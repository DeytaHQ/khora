"""Tests for ACL checker and enforcer."""

from uuid import uuid4

import pytest

from khora.acl.checker import ACLChecker, Permission, PermissionGrant, Principal
from khora.acl.enforcer import ACLContext, ACLEnforcer, ACLError


@pytest.mark.unit
class TestPermissionOrdering:
    """Permission enum supports comparison operators."""

    def test_read_less_than_write(self):
        assert Permission.READ < Permission.WRITE

    def test_write_less_than_admin(self):
        assert Permission.WRITE < Permission.ADMIN

    def test_admin_less_than_owner(self):
        assert Permission.ADMIN < Permission.OWNER

    def test_owner_is_highest(self):
        assert Permission.OWNER >= Permission.READ
        assert Permission.OWNER >= Permission.WRITE
        assert Permission.OWNER >= Permission.ADMIN

    def test_same_permission_equal(self):
        assert Permission.READ >= Permission.READ
        assert Permission.READ <= Permission.READ
        assert not (Permission.READ > Permission.READ)
        assert not (Permission.READ < Permission.READ)


@pytest.mark.unit
class TestPrincipal:
    """Principal factory methods."""

    def test_user_principal(self):
        p = Principal.user("alice")
        assert p.principal_type == "user"
        assert p.principal_id == "alice"

    def test_role_principal(self):
        p = Principal.role("admin")
        assert p.principal_type == "role"
        assert p.principal_id == "admin"

    def test_api_key_principal(self):
        p = Principal.api_key("key-123")
        assert p.principal_type == "api_key"
        assert p.principal_id == "key-123"

    def test_system_principal(self):
        p = Principal.system()
        assert p.principal_type == "system"
        assert p.principal_id == "system"


@pytest.mark.unit
class TestACLChecker:
    """ACLChecker grant, revoke, and check operations."""

    def test_direct_grant_allows_access(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)

        assert checker.check(user, "namespace", ns_id, Permission.READ) is True

    def test_no_grant_denies_access(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("bob")

        assert checker.check(user, "namespace", ns_id, Permission.READ) is False

    def test_higher_grant_satisfies_lower_requirement(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.ADMIN)

        assert checker.check(user, "namespace", ns_id, Permission.READ) is True
        assert checker.check(user, "namespace", ns_id, Permission.WRITE) is True

    def test_lower_grant_does_not_satisfy_higher_requirement(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)

        assert checker.check(user, "namespace", ns_id, Permission.WRITE) is False

    def test_system_principal_always_has_access(self):
        checker = ACLChecker()
        ns_id = uuid4()
        system = Principal.system()

        assert checker.check(system, "namespace", ns_id, Permission.OWNER) is True

    def test_grant_returns_permission_grant(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        grant = checker.grant(user, "namespace", ns_id, Permission.WRITE)

        assert isinstance(grant, PermissionGrant)
        assert grant.permission == Permission.WRITE
        assert grant.resource_id == ns_id

    def test_revoke_removes_grant(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)
        revoked = checker.revoke(user, "namespace", ns_id, Permission.READ)

        assert revoked == 1
        assert checker.check(user, "namespace", ns_id, Permission.READ) is False

    def test_revoke_all_permissions(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)
        checker.grant(user, "namespace", ns_id, Permission.WRITE)
        revoked = checker.revoke(user, "namespace", ns_id)

        assert revoked == 2
        assert checker.check(user, "namespace", ns_id, Permission.READ) is False

    def test_revoke_nonexistent_returns_zero(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("nobody")

        assert checker.revoke(user, "namespace", ns_id) == 0

    def test_grants_scoped_to_resource(self):
        checker = ACLChecker()
        ns1 = uuid4()
        ns2 = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns1, Permission.READ)

        assert checker.check(user, "namespace", ns1, Permission.READ) is True
        assert checker.check(user, "namespace", ns2, Permission.READ) is False

    def test_grants_scoped_to_principal(self):
        checker = ACLChecker()
        ns_id = uuid4()
        alice = Principal.user("alice")
        bob = Principal.user("bob")
        checker.grant(alice, "namespace", ns_id, Permission.WRITE)

        assert checker.check(alice, "namespace", ns_id, Permission.WRITE) is True
        assert checker.check(bob, "namespace", ns_id, Permission.WRITE) is False

    def test_get_effective_permissions(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)
        checker.grant(user, "namespace", ns_id, Permission.WRITE)

        effective = checker.get_effective_permissions(user, "namespace", ns_id)
        assert len(effective) == 2
        perms = {g.permission for g in effective}
        assert perms == {Permission.READ, Permission.WRITE}

    def test_list_grants_filter_by_principal(self):
        checker = ACLChecker()
        ns_id = uuid4()
        alice = Principal.user("alice")
        bob = Principal.user("bob")
        checker.grant(alice, "namespace", ns_id, Permission.READ)
        checker.grant(bob, "namespace", ns_id, Permission.WRITE)

        grants = checker.list_grants(principal=alice)
        assert len(grants) == 1
        assert grants[0].principal.principal_id == "alice"

    def test_list_grants_filter_by_resource_type(self):
        checker = ACLChecker()
        ns_id = uuid4()
        user = Principal.user("alice")
        checker.grant(user, "namespace", ns_id, Permission.READ)

        grants = checker.list_grants(resource_type="namespace")
        assert len(grants) == 1

        grants = checker.list_grants(resource_type="other")
        assert len(grants) == 0


@pytest.mark.unit
class TestACLEnforcer:
    """ACLEnforcer permission checking and enforcement."""

    def test_check_permission_granted(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("alice")
        enforcer.checker.grant(user, "namespace", ns_id, Permission.READ)
        ctx = ACLContext(principal=user, namespace_id=ns_id)

        assert enforcer.check_permission(ctx, "namespace", ns_id, Permission.READ) is True

    def test_check_permission_denied_raises_acl_error(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("bob")
        ctx = ACLContext(principal=user, namespace_id=ns_id)

        with pytest.raises(ACLError) as exc_info:
            enforcer.check_permission(ctx, "namespace", ns_id, Permission.READ)

        assert exc_info.value.principal == user
        assert exc_info.value.resource_type == "namespace"
        assert exc_info.value.resource_id == ns_id
        assert exc_info.value.required_permission == Permission.READ

    def test_disabled_enforcer_always_permits(self):
        enforcer = ACLEnforcer()
        enforcer.disable()
        ns_id = uuid4()
        user = Principal.user("nobody")
        ctx = ACLContext(principal=user)

        assert enforcer.check_permission(ctx, "namespace", ns_id, Permission.OWNER) is True
        assert enforcer.enabled is False

    def test_enable_re_enables_enforcement(self):
        enforcer = ACLEnforcer()
        enforcer.disable()
        enforcer.enable()

        assert enforcer.enabled is True

    def test_check_namespace_read(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("alice")
        enforcer.checker.grant(user, "namespace", ns_id, Permission.READ)
        ctx = ACLContext(principal=user)

        assert enforcer.check_namespace_read(ctx, ns_id) is True

    def test_check_namespace_write_denied(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("alice")
        enforcer.checker.grant(user, "namespace", ns_id, Permission.READ)
        ctx = ACLContext(principal=user)

        with pytest.raises(ACLError):
            enforcer.check_namespace_write(ctx, ns_id)

    def test_check_namespace_admin(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("alice")
        enforcer.checker.grant(user, "namespace", ns_id, Permission.ADMIN)
        ctx = ACLContext(principal=user)

        assert enforcer.check_namespace_admin(ctx, ns_id) is True

    def test_system_principal_bypasses_enforcer(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        system = Principal.system()
        ctx = ACLContext(principal=system)

        assert enforcer.check_permission(ctx, "namespace", ns_id, Permission.OWNER) is True

    def test_custom_checker_injected(self):
        checker = ACLChecker()
        enforcer = ACLEnforcer(checker=checker)
        assert enforcer.checker is checker


@pytest.mark.unit
class TestACLEnforcerRequireDecorator:
    """ACLEnforcer.require() decorator."""

    async def test_require_permits_authorized_call(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("alice")
        enforcer.checker.grant(user, "namespace", ns_id, Permission.WRITE)
        ctx = ACLContext(principal=user)

        @enforcer.require("namespace", Permission.WRITE)
        async def create_doc(context: ACLContext, namespace_id=None):
            return "ok"

        result = await create_doc(context=ctx, namespace_id=ns_id)
        assert result == "ok"

    async def test_require_blocks_unauthorized_call(self):
        enforcer = ACLEnforcer()
        ns_id = uuid4()
        user = Principal.user("bob")
        ctx = ACLContext(principal=user)

        @enforcer.require("namespace", Permission.WRITE)
        async def create_doc(context: ACLContext, namespace_id=None):
            return "ok"

        with pytest.raises(ACLError):
            await create_doc(context=ctx, namespace_id=ns_id)

    async def test_require_raises_value_error_on_missing_params(self):
        enforcer = ACLEnforcer()

        @enforcer.require("namespace", Permission.READ)
        async def some_func():
            return "ok"

        with pytest.raises(ValueError, match="Missing required parameter"):
            await some_func()
