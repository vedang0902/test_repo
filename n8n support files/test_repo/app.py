import time
import random
import logging
from prometheus_client import start_http_server, Counter, Gauge

# --------------------
# Logging setup
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# --------------------
# Prometheus metrics
# --------------------
ERROR_COUNT = Counter(
    "app_errors_total",
    "Total number of simulated application errors"
)

ERROR_RATE = Gauge(
    "app_error_rate",
    "Error rate indicator (1 = error, 0 = healthy)"
)

# --------------------
# App logic
# --------------------
def simulate_work():
    """
    Simulates application work.
    Randomly throws errors without stopping the app.
    """
    while True:
        try:
            time.sleep(5)  # app doing normal work

            # Inject fault every ~10–15 seconds
            if random.random() < 0.4:
                raise RuntimeError("Simulated pipeline failure")

            ERROR_RATE.set(0)
            logging.info("Service running normally")

        except Exception as e:
            ERROR_COUNT.inc()
            ERROR_RATE.set(1)

            logging.error(f"Application error occurred: {e}")

            # IMPORTANT: do NOT crash
            time.sleep(2)

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # Expose metrics
    start_http_server(8000, addr="0.0.0.0")
    logging.info("🚀 App running. Metrics exposed at :8000/metrics")

    simulate_work()
