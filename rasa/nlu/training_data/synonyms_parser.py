from typing import Text, List, Dict

from rasa.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
)


class SynonymsParser:
    @staticmethod
    def add_synonyms_from_entities(
        plain_text: Text, entities: List[Dict], existing_synonyms: Dict
    ) -> None:
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e[ENTITY_ATTRIBUTE_START] : e[ENTITY_ATTRIBUTE_END]]
            if e_text != e[ENTITY_ATTRIBUTE_VALUE]:
                SynonymsParser.add_synonym(
                    e_text, e[ENTITY_ATTRIBUTE_VALUE], existing_synonyms
                )

    @staticmethod
    def add_synonym(text: Text, value: Text, existing_synonyms: Dict) -> None:
        from rasa.nlu.training_data.util import check_duplicate_synonym

        check_duplicate_synonym(existing_synonyms, text, value, "reading markdown")
        existing_synonyms[text] = value
