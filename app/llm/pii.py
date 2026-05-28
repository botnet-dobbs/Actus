from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import structlog

log = structlog.get_logger()

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()


def scrub_pii(text: str, language: str = "en") -> tuple[str, bool]:
    results = _analyzer.analyze(text=text, language=language)
    if not results:
        return text, False
    log.warning(
        "pii_detected",
        entity_types=[r.entity_type for r in results],
        count=len(results),
    )
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, True
