import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.neo4j_connector import Neo4jConnector

def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a JSON dataset or .dump file into a selected Neo4j database"
    )
    parser.add_argument(
        "--instant_name",
        required=True,
        help="Name of the instance (e.g., 'cypherbench-db', 'mind-the-query-db')",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Target benchmark database",
    )
    parser.add_argument(
        "--filepath",
        required=True,
        help="Path to the JSON dataset file or .dump file",
    )
    parser.add_argument(
        "--host",
        default="neo4j://127.0.0.1",
        help="Neo4j host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7687,
        help="Neo4j port",
    )
    parser.add_argument(
        "--username",
        default="neo4j",
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        help="Neo4j password",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target database before import",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()


def build_connector(args):
    if args.instant_name == "cypherbench-db":
        DB_CONFIGS = {
            "company": "company",
            "fictional_character": "fictional.character",
            "flight_accident": "flight.accident",
            "geography": "geography",
            "movie": "movie",
            "nba": "nba",
            "politics": "politics",
        }
    else:
        raise ValueError(f"Unknown instant_name: '{args.instant_name}'")

    if args.db not in DB_CONFIGS:
        raise ValueError(
            f"Unknown database '{args.db}' for instance '{args.instant_name}'. "
            f"Valid options: {list(DB_CONFIGS.keys())}"
        )

    database_name = DB_CONFIGS[args.db]

    connector = Neo4jConnector(
        name=args.instant_name,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        database=database_name,
        debug=args.debug,
    )
    return connector, database_name


def main():
    args = parse_args()
    connector, database_name = build_connector(args)

    print(f"Starting import for database  : {args.db}")
    print(f"Resolved Neo4j database name  : {database_name}")
    print(f"Importing file                : {args.filepath}")

    if args.instant_name == "cypherbench-db":
        print("Import mode: JSON dataset")
        connector.import_json_dataset(
            filepath=args.filepath,
            db_name=database_name,
            overwrite=args.overwrite,
        )

    num_entities = connector.get_num_entities()
    num_relations = connector.get_num_relations()

    print("\n[SUCCESS] Data imported successfully!")
    print(f"Total nodes        : {num_entities}")
    print(f"Total relationships: {num_relations}")


if __name__ == "__main__":
    main()