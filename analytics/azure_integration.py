"""
AI Healthcare Data Platform – Azure Integration Guide.

Demonstrates integration patterns with Azure Data Factory (ADF) and
Azure Synapse Analytics for ingesting and transforming healthcare data
at enterprise scale.

This module provides:
- ADF pipeline trigger via REST API.
- Synapse Analytics SQL pool interaction via ODBC / SQLAlchemy.
- Helper utilities for Azure Blob Storage (ADLS Gen2).
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Environment-based configuration (never hard-code secrets)
# ---------------------------------------------------------------------------

TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "")
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg-healthcare")
ADF_FACTORY_NAME = os.environ.get("ADF_FACTORY_NAME", "adf-healthcare")
SYNAPSE_SERVER = os.environ.get("SYNAPSE_SERVER", "")          # e.g. myworkspace.sql.azuresynapse.net
SYNAPSE_DATABASE = os.environ.get("SYNAPSE_DATABASE", "healthcare_dw")
ADLS_ACCOUNT_NAME = os.environ.get("ADLS_ACCOUNT_NAME", "")
ADLS_CONTAINER = os.environ.get("ADLS_CONTAINER", "healthcare-data")


# ---------------------------------------------------------------------------
# Azure AD token helper
# ---------------------------------------------------------------------------


def get_azure_token(resource: str = "https://management.azure.com/") -> str:
    """
    Obtain an Azure AD OAuth2 bearer token using client-credentials flow.

    Requires AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET
    environment variables to be set.

    Args:
        resource: Azure resource URI to request a token for.

    Returns:
        Bearer token string.

    Raises:
        RuntimeError: If token acquisition fails.
    """
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "resource": resource,
    }
    response = requests.post(url, data=payload, timeout=30)
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Failed to obtain Azure AD token.")
    logger.info("Azure AD token acquired successfully.")
    return token


# ---------------------------------------------------------------------------
# Azure Data Factory
# ---------------------------------------------------------------------------


def trigger_adf_pipeline(
    pipeline_name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Trigger an Azure Data Factory pipeline run via the ADF REST API.

    Args:
        pipeline_name: Name of the ADF pipeline to run.
        parameters: Optional dict of pipeline parameters.

    Returns:
        API response JSON containing the run ID.
    """
    token = get_azure_token()
    url = (
        f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
        f"/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.DataFactory/factories/{ADF_FACTORY_NAME}"
        f"/pipelines/{pipeline_name}/createRun?api-version=2018-06-01"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"parameters": parameters or {}}

    response = requests.post(url, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info("ADF pipeline '%s' triggered – run ID: %s", pipeline_name, result.get("runId"))
    return result


def get_adf_pipeline_run_status(run_id: str) -> Dict[str, Any]:
    """
    Poll the status of an ADF pipeline run.

    Args:
        run_id: Run ID returned by trigger_adf_pipeline.

    Returns:
        API response JSON with status and timing information.
    """
    token = get_azure_token()
    url = (
        f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
        f"/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.DataFactory/factories/{ADF_FACTORY_NAME}"
        f"/pipelineRuns/{run_id}?api-version=2018-06-01"
    )
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info("ADF run '%s' status: %s", run_id, result.get("status"))
    return result


# ---------------------------------------------------------------------------
# Azure Synapse Analytics
# ---------------------------------------------------------------------------


def get_synapse_connection():
    """
    Return a SQLAlchemy engine connected to Azure Synapse Analytics SQL pool.

    Requires pyodbc and the Microsoft ODBC Driver 18 for SQL Server to be
    installed on the host machine.

    Returns:
        SQLAlchemy engine.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError("Install sqlalchemy: pip install sqlalchemy pyodbc") from exc

    connection_string = (
        f"mssql+pyodbc://{CLIENT_ID}:{CLIENT_SECRET}@{SYNAPSE_SERVER}:1433"
        f"/{SYNAPSE_DATABASE}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&Authentication=ActiveDirectoryServicePrincipal"
        "&Encrypt=yes&TrustServerCertificate=no"
    )
    engine = create_engine(connection_string, fast_executemany=True)
    logger.info("Synapse Analytics connection engine created for database '%s'.", SYNAPSE_DATABASE)
    return engine


def load_dataframe_to_synapse(
    df,
    table_name: str,
    schema: str = "dbo",
    if_exists: str = "append",
) -> None:
    """
    Write a pandas DataFrame to an Azure Synapse dedicated SQL pool table.

    Args:
        df: DataFrame to load.
        table_name: Target table name.
        schema: Target schema (default 'dbo').
        if_exists: 'append' or 'replace'.
    """
    engine = get_synapse_connection()
    df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False, chunksize=1000)
    logger.info(
        "Loaded %d rows to Synapse table [%s].[%s]", len(df), schema, table_name
    )


# ---------------------------------------------------------------------------
# Azure Data Lake Storage Gen2 (ADLS)
# ---------------------------------------------------------------------------


def upload_to_adls(local_path: str, blob_path: str) -> str:
    """
    Upload a local file to Azure Data Lake Storage Gen2.

    Requires the azure-storage-blob package:
        pip install azure-storage-blob azure-identity

    Args:
        local_path: Path to the local file.
        blob_path: Destination path within the container.

    Returns:
        Full ADLS URL of the uploaded blob.
    """
    try:
        from azure.identity import ClientSecretCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise ImportError(
            "Install azure-storage-blob and azure-identity: "
            "pip install azure-storage-blob azure-identity"
        ) from exc

    credential = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
    account_url = f"https://{ADLS_ACCOUNT_NAME}.dfs.core.windows.net"
    client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = client.get_blob_client(container=ADLS_CONTAINER, blob=blob_path)

    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    url = f"{account_url}/{ADLS_CONTAINER}/{blob_path}"
    logger.info("Uploaded '%s' → %s", local_path, url)
    return url
