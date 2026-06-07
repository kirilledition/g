---
name: linear
description: Use Linear MCP tools for Linear reads, comments, labels, generated issues, state transitions, and attachments during Symphony sessions.
---

# Linear

Use this skill during Symphony sessions when Linear issue comments, state
transitions, labels, generated issues, project links, or attachments need to be
read or updated.

## Tool Priority

1. Prefer the Linear MCP tools configured for Codex. Use them for issue lookup,
   issue search, issue create/update, comments, labels, attachments, and state
   transitions.
2. If Linear MCP is unavailable in the current session, use Symphony's
   `linear_graphql` client tool if exposed. It reuses Symphony's configured
   Linear authentication.
3. Do not use raw HTTP requests to Linear unless both MCP and `linear_graphql`
   are unavailable and the issue is blocked without Linear access.

## Generated Issues

Only issues labeled `task-generator` may create new issues that include the
`symphony` dispatch label.

When creating generated issues:

- Search existing Linear issues first to avoid duplicates.
- Create at most 5 `symphony` issues from one parent unless the parent issue
  explicitly states a different limit.
- Add the `generated` label to all agent-created follow-up issues.
- Add `symphony` only when the task is concrete, bounded, and ready for
  unattended implementation.
- Do not add `task-generator` to generated issues unless the parent explicitly
  asks for recursive task generation.
- Include the parent issue identifier and URL in the generated issue
  description.
- Use labels such as `cpu`, `gpu`, `benchmark`, and `data` for routing hints.

Generated issue descriptions must use this shape:

```text
Parent: <issue identifier and URL>

Background:
<What was found and why it matters.>

Scope:
<Exactly what should change.>

Acceptance criteria:
- <Concrete expected outcome.>

Validation:
- <Specific local, SLURM, benchmark, or CI command.>

Non-goals:
- <What this issue should not touch.>
```

## GraphQL Fallback

Tool input:

```json
{
  "query": "query or mutation document",
  "variables": {
    "optional": "GraphQL variables"
  }
}
```

Treat a top-level `errors` array as a failed operation even when the tool call
itself succeeds.

## Common Queries

Lookup an issue by key:

```graphql
query IssueByKey($key: String!) {
  issue(id: $key) {
    id
    identifier
    title
    description
    url
    branchName
    state {
      id
      name
      type
    }
    team {
      id
      key
      name
      states {
        nodes {
          id
          name
          type
        }
      }
    }
    project {
      id
      name
    }
    comments {
      nodes {
        id
        body
      }
    }
    attachments {
      nodes {
        id
        title
        url
        sourceType
      }
    }
  }
}
```

Search for likely duplicate issues before generating follow-ups:

```graphql
query SearchIssues($teamId: String!, $text: String!) {
  issues(
    first: 20
    filter: {
      team: { id: { eq: $teamId } }
      or: [
        { title: { containsIgnoreCase: $text } }
        { description: { containsIgnoreCase: $text } }
      ]
    }
  ) {
    nodes {
      id
      identifier
      title
      url
      state {
        name
      }
      labels {
        nodes {
          name
        }
      }
    }
  }
}
```

Resolve team states and labels before creating a generated issue:

```graphql
query TeamRouting($teamId: String!) {
  team(id: $teamId) {
    id
    key
    states {
      nodes {
        id
        name
      }
    }
    labels {
      nodes {
        id
        name
      }
    }
  }
}
```

Create a generated follow-up issue:

```graphql
mutation CreateGeneratedIssue($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      title
      url
      state {
        name
      }
      labels {
        nodes {
          name
        }
      }
    }
  }
}
```

Use `IssueCreateInput.parentId` for the parent issue ID when the follow-up is a
child of the current issue. Use `labelIds` for `generated` and, only when ready
for unattended implementation, `symphony`.

Update an issue state after resolving the target `stateId` from the issue team:

```graphql
mutation UpdateIssueState($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) {
    success
    issue {
      id
      identifier
      state {
        id
        name
      }
    }
  }
}
```

Update an issue description, for example to maintain `## Agent Learnings`:

```graphql
mutation UpdateIssueDescription($id: String!, $description: String!) {
  issueUpdate(id: $id, input: { description: $description }) {
    success
    issue {
      id
      identifier
      description
    }
  }
}
```

Create a workpad comment:

```graphql
mutation CreateComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment {
      id
      body
    }
  }
}
```

Update a workpad comment:

```graphql
mutation UpdateComment($id: String!, $body: String!) {
  commentUpdate(id: $id, input: { body: $body }) {
    success
    comment {
      id
      body
    }
  }
}
```

Create a GitHub attachment for a pushed branch, main commit, or explicitly
requested PR:

```graphql
mutation CreateAttachment($issueId: String!, $title: String!, $url: String!) {
  attachmentCreate(input: { issueId: $issueId, title: $title, url: $url }) {
    success
    attachment {
      id
      title
      url
    }
  }
}
```
