import os
from typing import Optional
import logging

logger = logging.getLogger('atwz.security')


def read_api_keys_from_env() -> dict:
    # Try keyring first
    result = {}
    try:
        import keyring
        result['KITE_API_KEY'] = keyring.get_password('AlgoTradingWithZerodha', 'KITE_API_KEY')
        result['KITE_API_SECRET'] = keyring.get_password('AlgoTradingWithZerodha', 'KITE_API_SECRET')
        result['SMTP_USER'] = keyring.get_password('AlgoTradingWithZerodha', 'SMTP_USER')
        result['SMTP_PASS'] = keyring.get_password('AlgoTradingWithZerodha', 'SMTP_PASS')
    except Exception:
        # keyring not available or keys not set
        result['KITE_API_KEY'] = None
        result['KITE_API_SECRET'] = None
        result['SMTP_USER'] = None
        result['SMTP_PASS'] = None

    # fallback to environment
    result['KITE_API_KEY'] = result.get('KITE_API_KEY') or os.getenv('KITE_API_KEY')
    result['KITE_API_SECRET'] = result.get('KITE_API_SECRET') or os.getenv('KITE_API_SECRET')
    result['SMTP_USER'] = result.get('SMTP_USER') or os.getenv('SMTP_USER')
    result['SMTP_PASS'] = result.get('SMTP_PASS') or os.getenv('SMTP_PASS')

    # optional: check HashiCorp Vault if configured
    vault_addr = os.getenv('VAULT_ADDR')
    vault_token = os.getenv('VAULT_TOKEN')
    vault_path = os.getenv('VAULT_PATH')
    if vault_addr and vault_token and vault_path:
        vault_data = get_secret_vault(vault_addr, vault_token, vault_path)
        if isinstance(vault_data, dict):
            result['KITE_API_KEY'] = result.get('KITE_API_KEY') or vault_data.get('KITE_API_KEY')
            result['KITE_API_SECRET'] = result.get('KITE_API_SECRET') or vault_data.get('KITE_API_SECRET')
            result['SMTP_USER'] = result.get('SMTP_USER') or vault_data.get('SMTP_USER')
            result['SMTP_PASS'] = result.get('SMTP_PASS') or vault_data.get('SMTP_PASS')

    return result

def validate_symbol(symbol: str) -> bool:
    if not symbol or not isinstance(symbol, str):
        return False
    if len(symbol) > 64:
        return False
    return True


def set_secret_keyring(key: str, value: str):
    try:
        import keyring
        keyring.set_password('AlgoTradingWithZerodha', key, value)
        return True
    except Exception as e:
        logger.exception('Failed to set keyring secret')
        return False


def get_secret_keyring(key: str) -> Optional[str]:
    try:
        import keyring
        return keyring.get_password('AlgoTradingWithZerodha', key)
    except Exception:
        logger.exception('Failed to get keyring secret')
        return None


def get_secret_vault(vault_addr: str, token: str, path: str) -> Optional[dict]:
    """Example: read secret from HashiCorp Vault using hvac. Returns dict or None.

    This is an example helper; production usage should handle TLS, renewals, and error handling.
    """
    try:
        import hvac
    except Exception:
        logger.warning('hvac not installed')
        return None
    try:
        client = hvac.Client(url=vault_addr, token=token)
        secret = client.secrets.kv.v2.read_secret_version(path=path)
        return secret.get('data', {}).get('data')
    except Exception:
        logger.exception('Failed to read secret from vault')
        return None
