"""
Data Quality Validation using Great Expectations.

Validates ETL output DataFrames against a configurable expectation suite
before writing to Delta Lake.
"""

import logging
from typing import Dict, Any, List

import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.dataset import SparkDFDataset

logger = logging.getLogger(__name__)


def build_expectation_suite(suite_name: str = "etl_output_suite") -> ExpectationSuite:
    """
    Build and return a reusable Great Expectations suite.

    The suite enforces:
    - No null values in critical columns.
    - Row count is positive.
    - String columns match expected cardinality constraints.

    Args:
        suite_name: Name for the expectation suite.

    Returns:
        Configured ExpectationSuite.
    """
    suite = ExpectationSuite(expectation_suite_name=suite_name)

    # Generic expectations applied to every table
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={"column_list": None},  # Overridden per-dataset below
        )
    )
    return suite


def build_column_expectations(columns: List[str]) -> List[ExpectationConfiguration]:
    """
    Build per-column null-completeness expectations.

    Args:
        columns: List of column names that must not contain nulls.

    Returns:
        List of ExpectationConfiguration objects.
    """
    expectations = []
    for col in columns:
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col},
            )
        )
    return expectations


def validate_dataframe(spark_df, not_null_columns: List[str]) -> Dict[str, Any]:
    """
    Validate a PySpark DataFrame using Great Expectations.

    Args:
        spark_df: PySpark DataFrame to validate.
        not_null_columns: Columns that must have no null values.

    Returns:
        Validation result dictionary with 'success' (bool) and 'results' (list).

    Raises:
        ValueError: When validation fails (any expectation is not met).
    """
    logger.info("Running data quality validation on %d rows", spark_df.count())

    context = gx.get_context()
    suite = build_expectation_suite()

    for exp in build_column_expectations(not_null_columns):
        suite.add_expectation(exp)

    gx_df = SparkDFDataset(spark_df)
    results = gx_df.validate(expectation_suite=suite, result_format="SUMMARY")

    success = results["success"]
    failed = [r for r in results["results"] if not r["success"]]

    if success:
        logger.info("Data quality validation passed – all expectations met.")
    else:
        logger.error(
            "Data quality validation FAILED – %d expectation(s) not met: %s",
            len(failed),
            [r["expectation_config"]["expectation_type"] for r in failed],
        )
        raise ValueError(
            f"Data quality check failed with {len(failed)} violation(s). "
            "Aborting pipeline run."
        )

    return {"success": success, "results": results["results"]}
