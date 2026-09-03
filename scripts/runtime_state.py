import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import json


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runtime"
STATE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = STATE_DIR / "riskpulse_state.db"


class RiskState:

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self._initialize_database()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self):

        conn = self._connect()
        cur = conn.cursor()

        # ---------------------------------------------------------
        # Customer state
        # ---------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                transaction_count INTEGER DEFAULT 0,
                total_amount REAL DEFAULT 0,
                max_amount REAL DEFAULT 0,
                previous_timestamp TEXT,
                created_at TEXT
            )
        """)

        # ---------------------------------------------------------
        # Merchant state
        # ---------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS merchants (
                merchant_id TEXT PRIMARY KEY,
                transaction_count INTEGER DEFAULT 0,
                total_amount REAL DEFAULT 0,
                previous_amount REAL,
                previous_timestamp TEXT,
                fraud_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        # ---------------------------------------------------------
        # Transaction history
        # Used for velocity and relationship calculations.
        # ---------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                customer_id TEXT,
                merchant_id TEXT,
                device_id TEXT,
                ip_id TEXT,
                shipping_address_id TEXT,
                billing_address_id TEXT,
                amount REAL,
                is_fraud INTEGER
            )
        """)
        # ---------------------------------------------------------
        # Address columns migration
        # ---------------------------------------------------------
        # Existing databases created before address tracking may
        # not have these columns.
        # SQLite safely ignores this migration if they already exist.
        # ---------------------------------------------------------

        existing_columns = {
            row["name"]
            for row in cur.execute(
                "PRAGMA table_info(transactions)"
            ).fetchall()
        }

        if "shipping_address_id" not in existing_columns:
            cur.execute(
                """
                ALTER TABLE transactions
                ADD COLUMN shipping_address_id TEXT
                """
            )

        if "billing_address_id" not in existing_columns:
            cur.execute(
                """
                ALTER TABLE transactions
                ADD COLUMN billing_address_id TEXT
                """
            )

        # ---------------------------------------------------------
        # Entity relationships
        # ---------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entity_relationships (
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                related_type TEXT NOT NULL,
                related_id TEXT NOT NULL,
                first_seen TEXT,
                last_seen TEXT,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (
                    entity_type,
                    entity_id,
                    related_type,
                    related_id
                )
            )
        """)

        conn.commit()
        conn.close()

    # =============================================================
    # CUSTOMER
    # =============================================================

    def get_customer_state(self, customer_id):

        conn = self._connect()
        row = conn.execute(
            """
            SELECT *
            FROM customers
            WHERE customer_id = ?
            """,
            (customer_id,)
        ).fetchone()

        conn.close()

        if row is None:
            return {
                "exists": False,
                "transaction_count": 0,
                "total_amount": 0.0,
                "avg_amount": 0.0,
                "max_amount": 0.0,
                "previous_timestamp": None
            }

        count = row["transaction_count"]

        return {
            "exists": True,
            "transaction_count": count,
            "total_amount": float(row["total_amount"]),
            "avg_amount": (
                float(row["total_amount"]) / count
                if count > 0 else 0.0
            ),
            "max_amount": float(row["max_amount"]),
            "previous_timestamp": row["previous_timestamp"]
        }

    # =============================================================
    # MERCHANT
    # =============================================================

    def get_merchant_state(self, merchant_id):

        conn = self._connect()
        row = conn.execute(
            """
            SELECT *
            FROM merchants
            WHERE merchant_id = ?
            """,
            (merchant_id,)
        ).fetchone()

        conn.close()

        if row is None:
            return {
                "exists": False,
                "transaction_count": 0,
                "total_amount": 0.0,
                "avg_amount": 0.0,
                "previous_amount": None,
                "previous_timestamp": None,
                "fraud_count": 0
            }

        count = row["transaction_count"]

        return {
            "exists": True,
            "transaction_count": count,
            "total_amount": float(row["total_amount"]),
            "avg_amount": (
                float(row["total_amount"]) / count
                if count > 0 else 0.0
            ),
            "previous_amount": row["previous_amount"],
            "previous_timestamp": row["previous_timestamp"],
            "fraud_count": int(row["fraud_count"])
        }

    # =============================================================
    # RELATIONSHIP LOOKUPS
    # =============================================================

    def get_related_count(
        self,
        entity_type,
        entity_id,
        related_type
    ):

        conn = self._connect()

        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM entity_relationships
            WHERE entity_type = ?
              AND entity_id = ?
              AND related_type = ?
            """,
            (
                entity_type,
                entity_id,
                related_type
            )
        ).fetchone()

        conn.close()

        return int(row["count"])

    def get_related_ids(
        self,
        entity_type,
        entity_id,
        related_type
    ):

        conn = self._connect()

        rows = conn.execute(
            """
            SELECT related_id
            FROM entity_relationships
            WHERE entity_type = ?
              AND entity_id = ?
              AND related_type = ?
            """,
            (
                entity_type,
                entity_id,
                related_type
            )
        ).fetchall()

        conn.close()

        return [row["related_id"] for row in rows]

    # =============================================================
    # VELOCITY
    # =============================================================

    def get_transaction_count_since(
        self,
        entity_field,
        entity_id,
        timestamp,
        hours
    ):

        allowed_fields = {
            "customer_id",
            "merchant_id",
            "device_id",
            "ip_id"
        }

        if entity_field not in allowed_fields:
            raise ValueError("Invalid entity field")

        current_time = datetime.fromisoformat(timestamp)
        cutoff = current_time - timedelta(hours=hours)

        conn = self._connect()

        query = f"""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE {entity_field} = ?
              AND timestamp >= ?
              AND timestamp < ?
        """

        row = conn.execute(
            query,
            (
                entity_id,
                cutoff.isoformat(),
                timestamp
            )
        ).fetchone()

        conn.close()

        return int(row["count"])

    # =============================================================
    # ADDRESS HISTORY
    # =============================================================

    def get_address_transaction_count_before(
        self,
        address_field,
        address_id,
        timestamp,
    ):
        """
        Count all transactions for an address strictly before
        the current transaction timestamp.

        This mirrors the behavioral dataset's:
            groupby(address).cumcount()
            - same-timestamp transactions

        Therefore transactions at the exact same timestamp are
        excluded.
        """

        allowed_fields = {
            "shipping_address_id",
            "billing_address_id",
        }

        if address_field not in allowed_fields:
            raise ValueError("Invalid address field")

        if not address_id:
            return 0

        conn = self._connect()

        row = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE {address_field} = ?
              AND timestamp < ?
            """,
            (
                address_id,
                timestamp,
            ),
        ).fetchone()

        conn.close()

        return int(row["count"])

    def get_address_transaction_count_since(
        self,
        address_field,
        address_id,
        timestamp,
        hours,
    ):
        """
        Count strictly earlier transactions for an address inside
        the requested time window.

        Mirrors the behavioral generator's prior-window logic:
            current_time - window <= transaction_time < current_time
        """

        allowed_fields = {
            "shipping_address_id",
            "billing_address_id",
        }

        if address_field not in allowed_fields:
            raise ValueError("Invalid address field")

        if not address_id:
            return 0

        current_time = datetime.fromisoformat(timestamp)
        cutoff = current_time - timedelta(hours=hours)

        conn = self._connect()

        row = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE {address_field} = ?
              AND timestamp >= ?
              AND timestamp < ?
            """,
            (
                address_id,
                cutoff.isoformat(),
                timestamp,
            ),
        ).fetchone()

        conn.close()

        return int(row["count"])

    def get_address_unique_count_before(
        self,
        address_field,
        address_id,
        related_field,
        timestamp,
    ):
        """
        Count unique customers/merchants previously associated with
        an address.

        The current transaction is excluded.

        This mirrors the generator's _prior_unique_count_by_key()
        semantics.
        """

        allowed_addresses = {
            "shipping_address_id",
            "billing_address_id",
        }

        allowed_related = {
            "customer_id",
            "merchant_id",
        }

        if address_field not in allowed_addresses:
            raise ValueError("Invalid address field")

        if related_field not in allowed_related:
            raise ValueError("Invalid related field")

        if not address_id:
            return 0

        conn = self._connect()

        row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT {related_field}) AS count
            FROM transactions
            WHERE {address_field} = ?
              AND timestamp < ?
              AND {related_field} IS NOT NULL
            """,
            (
                address_id,
                timestamp,
            ),
        ).fetchone()

        conn.close()

        return int(row["count"])
    # =============================================================
    # UPDATE STATE
    # IMPORTANT:
    # This must only happen AFTER scoring.
    # =============================================================

    def update_after_transaction(self, tx, fraud_label=None):

        timestamp = tx["timestamp"]
        amount = float(tx["transaction_amount"])

        customer_id = tx.get("customer_id")
        merchant_id = tx.get("merchant_id")
        device_id = tx.get("device_id")
        ip_id = tx.get("ip_id")
        transaction_id = tx["transaction_id"]

        conn = self._connect()
        cur = conn.cursor()

        # ---------------------------------------------------------
        # Customer
        # ---------------------------------------------------------

        if customer_id:

            existing = cur.execute(
                """
                SELECT *
                FROM customers
                WHERE customer_id = ?
                """,
                (customer_id,)
            ).fetchone()

            if existing is None:

                cur.execute(
                    """
                    INSERT INTO customers
                    (
                        customer_id,
                        transaction_count,
                        total_amount,
                        max_amount,
                        previous_timestamp,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        customer_id,
                        1,
                        amount,
                        amount,
                        timestamp,
                        timestamp
                    )
                )

            else:

                cur.execute(
                    """
                    UPDATE customers
                    SET transaction_count = transaction_count + 1,
                        total_amount = total_amount + ?,
                        max_amount = MAX(max_amount, ?),
                        previous_timestamp = ?
                    WHERE customer_id = ?
                    """,
                    (
                        amount,
                        amount,
                        timestamp,
                        customer_id
                    )
                )

        # ---------------------------------------------------------
        # Merchant
        # ---------------------------------------------------------

        if merchant_id:

            existing = cur.execute(
                """
                SELECT *
                FROM merchants
                WHERE merchant_id = ?
                """,
                (merchant_id,)
            ).fetchone()

            fraud_increment = (
                1 if fraud_label == 1 else 0
            )

            if existing is None:

                cur.execute(
                    """
                    INSERT INTO merchants
                    (
                        merchant_id,
                        transaction_count,
                        total_amount,
                        previous_amount,
                        previous_timestamp,
                        fraud_count,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        merchant_id,
                        1,
                        amount,
                        amount,
                        timestamp,
                        fraud_increment,
                        timestamp
                    )
                )

            else:

                cur.execute(
                    """
                    UPDATE merchants
                    SET transaction_count = transaction_count + 1,
                        total_amount = total_amount + ?,
                        previous_amount = ?,
                        previous_timestamp = ?,
                        fraud_count = fraud_count + ?
                    WHERE merchant_id = ?
                    """,
                    (
                        amount,
                        amount,
                        timestamp,
                        fraud_increment,
                        merchant_id
                    )
                )

        # ---------------------------------------------------------
        # Transaction history
        # ---------------------------------------------------------

        cur.execute(
            """
            INSERT OR IGNORE INTO transactions
            (
                transaction_id,
                timestamp,
                customer_id,
                merchant_id,
                device_id,
                ip_id,
                shipping_address_id,
                billing_address_id,
                amount,
                is_fraud
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                timestamp,
                customer_id,
                merchant_id,
                device_id,
                ip_id,
                tx.get("shipping_address_id"),
                tx.get("billing_address_id"),
                amount,
                fraud_label,
            )
        )

        # ---------------------------------------------------------
        # Relationships
        # ---------------------------------------------------------

        relationships = [
            # Customer relationships
            ("customer", customer_id, "merchant", merchant_id),
            ("customer", customer_id, "device", device_id),
            ("customer", customer_id, "ip", ip_id),
            ("customer", customer_id, "shipping_address",
            tx.get("shipping_address_id")),
            ("customer", customer_id, "billing_address",
            tx.get("billing_address_id")),

            # Merchant relationships
            ("merchant", merchant_id, "customer", customer_id),
            ("merchant", merchant_id, "device", device_id),
            ("merchant", merchant_id, "ip", ip_id),
            ("merchant", merchant_id, "shipping_address",
            tx.get("shipping_address_id")),
            ("merchant", merchant_id, "billing_address",
            tx.get("billing_address_id")),

            # IP relationships
            ("ip", ip_id, "customer", customer_id),
            ("ip", ip_id, "merchant", merchant_id),

            # Device relationships
            ("device", device_id, "customer", customer_id),

            # Address relationships
            ("shipping_address",
            tx.get("shipping_address_id"),
            "customer",
            customer_id),

            ("shipping_address",
            tx.get("shipping_address_id"),
            "merchant",
            merchant_id),

            ("billing_address",
            tx.get("billing_address_id"),
            "customer",
            customer_id),

            ("billing_address",
            tx.get("billing_address_id"),
            "merchant",
            merchant_id),
        ]

        for entity_type, entity_id, related_type, related_id in relationships:

            if not entity_id or not related_id:
                continue

            cur.execute(
                """
                INSERT INTO entity_relationships
                (
                    entity_type,
                    entity_id,
                    related_type,
                    related_id,
                    first_seen,
                    last_seen,
                    count
                )
                VALUES (?, ?, ?, ?, ?, ?, 1)

                ON CONFLICT(
                    entity_type,
                    entity_id,
                    related_type,
                    related_id
                )
                DO UPDATE SET
                    last_seen = excluded.last_seen,
                    count = count + 1
                """,
                (
                    entity_type,
                    entity_id,
                    related_type,
                    related_id,
                    timestamp,
                    timestamp
                )
            )

        conn.commit()
        conn.close()


if __name__ == "__main__":

    state = RiskState()

    print("RiskPulse runtime state initialized.")
    print(f"Database: {DB_PATH}")