import os

loginURL = {
    "baseURL": "https://test-user-oneid.deep-time.org/ngiam-rst/v1/sdk/login/sso",
    "appCode": os.getenv("APP_CODE"),
    "secretCode": os.getenv("SECRET_CODE"),
    "tokenChange": "https://test-user-oneid.deep-time.org/ngiam-rst/oauth2/token",
    "validate": "https://test-user-oneid.deep-time.org/ngiam-rst/oauth2/validate",
    "infoChange": "https://test-user-oneid.deep-time.org/ngiam-rst/oauth2/userinfo",
    "redirectUrl": "https://deep-time.org/"
}

# Ensure all required environment variables are set
required_env_vars = ["APP_CODE", "SECRET_CODE"]
missing_vars = [var for var in required_env_vars if not loginURL[var.lower()]]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
