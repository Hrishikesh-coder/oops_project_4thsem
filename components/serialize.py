import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, List


class ResultSerializer:
    """Serializes classifier outputs into JSON-friendly records."""

    @staticmethod
    def to_records(results: Iterable[Any]) -> List[dict]:
        records: List[dict] = []
        for item in results:
            if is_dataclass(item):
                records.append(asdict(item))
            elif isinstance(item, dict):
                records.append(item)
            else:
                raise TypeError(
                    "Unsupported result type for serialization: "
                    f"{type(item).__name__}"
                )
        return records

    @staticmethod
    def to_json(results: Iterable[Any], indent: int = 4) -> str:
        return json.dumps(ResultSerializer.to_records(results), indent=indent)

    @staticmethod
    def write_json(results: Iterable[Any], output_path: str, indent: int = 4) -> None:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(ResultSerializer.to_json(results, indent=indent))
