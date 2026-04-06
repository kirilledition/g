## 2025-02-12 - [Python/Mocks] Mock test failures due to missing module attribute
**Learning:** Mocking targets with `patch` will fail with an `AttributeError` if the attribute doesn't exist on the target module, which happens often during refactoring.
**Action:** When a mock test fails with "module does not have the attribute", trace where the attribute actually resides now and update the patch string. For example, `g.cli.run_linear_api` was refactored to just `g.api.linear`.
