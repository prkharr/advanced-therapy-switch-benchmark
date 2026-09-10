"""Export reviewed Snowflake SELECTs into the raw folder used by run_pipeline.py."""

from therapy_switch.delivery.raw_export import main

if __name__ == "__main__":
    raise SystemExit(main())
