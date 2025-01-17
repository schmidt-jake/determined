# Database Migrations

We use `go-pg/migrations`: https://github.com/go-pg/migrations

## Running migrations manually

```bash
determined-master [MASTER_ARGS] migrate [MIGRATION_ARGS]
```

where `MASTER_ARGS` are the normal determined-master command flags,
and `MIGRATION_ARGS` are `go-pg/migrations` args.

For example, to migrate down to a specific version, run

```bash
determined-master --config-file /path/to/master.yaml migrate down 20210917133742
```

## Creating new migrations

We use timestamps instead of sequential numbers, standard for `go-pg/migrations`.
When creating a new migration, either write it manually, or use this script:

```bash

touch $(date +%Y%m%d%H%M%S)_migration_name.{up,down}.sql
```
