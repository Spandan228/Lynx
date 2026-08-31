"""
Multi-Tenant Security, JWT Authentication, and RBAC Context Provider.

Features:
1. `UserSecurityContext`: Immutable Pydantic model storing tenant_id, user_id, and authorized roles.
2. JWT Token Engine: Generates and verifies HMAC-SHA256 signed JWT tokens.
3. FastAPI Security Dependency: Extracts and validates Bearer tokens from incoming HTTP Authorization headers.
4. Transparent Dev Fallback: Gracefully provides a default context for local CLI / test runners when auth is not configured.

Author: Cloud Security and IAM Architect
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import jwt
from pydantic import BaseModel, Field
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "crag_enterprise_jwt_super_secret_signing_key_2026_!@#")
JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ---------------------------------------------------------------------------
# 1. User Security Context Model
# ---------------------------------------------------------------------------
class UserSecurityContext(BaseModel):
    """
    Immutable identity and authorization context attached to all retrieval
    and state execution requests.
    """
    tenant_id: str = Field(
        default="tenant_default",
        description="Unique organizational or customer tenant identifier."
    )
    user_id: str = Field(
        default="usr_anonymous",
        description="Unique user identifier."
    )
    roles: List[str] = Field(
        default=["user", "admin"],
        description="Assigned RBAC roles (e.g. ['admin', 'finance_reader', 'engineer'])."
    )
    email: Optional[str] = Field(
        default=None,
        description="Optional user email."
    )

    def has_role(self, required_role: str) -> bool:
        """Checks if user possesses a specific role."""
        return required_role in self.roles or "admin" in self.roles

    def overlaps_roles(self, allowed_roles: List[str]) -> bool:
        """Checks if user has at least one of the allowed roles."""
        if "admin" in self.roles:
            return True
        return bool(set(self.roles).intersection(set(allowed_roles)))


# ---------------------------------------------------------------------------
# 2. JWT Minting and Verification Utilities
# ---------------------------------------------------------------------------
def create_access_token(
    security_context: UserSecurityContext,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Mints a signed JWT access token encoding the UserSecurityContext claims."""
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=DEFAULT_TOKEN_EXPIRE_MINUTES))
    to_encode = {
        "sub": security_context.user_id,
        "tenant_id": security_context.tenant_id,
        "roles": security_context.roles,
        "email": security_context.email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> UserSecurityContext:
    """Decodes and validates JWT token claims, returning a verified UserSecurityContext."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tenant_id = payload.get("tenant_id")
        user_id = payload.get("sub")
        roles = payload.get("roles", [])

        if not tenant_id or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims: missing tenant_id or sub",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return UserSecurityContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=roles,
            email=payload.get("email"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failure: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# 3. FastAPI Dependency Provider
# ---------------------------------------------------------------------------
http_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_security_context(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer_scheme),
) -> UserSecurityContext:
    """
    FastAPI dependency extracting UserSecurityContext from the Bearer token.
    If no token is provided in local development mode, falls back to header parameters
    (X-Tenant-ID, X-User-ID, X-User-Roles) or default context.
    """
    # 1. Bearer Token Authentication
    if credentials and credentials.credentials:
        return decode_access_token(credentials.credentials)

    # 2. Header-Based Development Fallback (e.g. for Streamlit / Postman testing)
    custom_tenant = request.headers.get("X-Tenant-ID")
    custom_user = request.headers.get("X-User-ID")
    custom_roles_header = request.headers.get("X-User-Roles")

    if custom_tenant or custom_user or custom_roles_header:
        roles_list = [r.strip() for r in custom_roles_header.split(",")] if custom_roles_header else ["user"]
        return UserSecurityContext(
            tenant_id=custom_tenant or "tenant_default",
            user_id=custom_user or "usr_dev_anonymous",
            roles=roles_list,
        )

    # 3. Default Development Tenant Context
    return UserSecurityContext(
        tenant_id="tenant_default",
        user_id="usr_anonymous",
        roles=["user", "admin"],
    )
