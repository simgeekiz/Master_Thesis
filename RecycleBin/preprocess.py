"""Data enrichment and preparation module.

This module creates FoLiA XML files with the given data. It enriches the XML files
with part-of-speech, lemma, named entity and time tags.

For more on FoLiA (Format for Linguistic Annotations) please see
`FoLiA :: Format for Linguistic Annotation <https://proycon.github.io/folia/>`_.
"""

import os
import logging
from random import randint
from datetime import datetime
import xml.etree.ElementTree as ET
from subprocess import Popen, PIPE
from dateutil import parser
from pandas import DataFrame

from pynlpl.formats import folia

# from .settings import EXTERNAL_PACKAGES, TMP_FOLDER_PATH, TAGSETS
# from .utils import nlp_tools
# from .utils import folia_processing as fp
# from .utils.data_processing import convert_temporal, get_unique_sentence_ids
# from .feature_extraction import get_document_token_features
# from . import postprocessing as postp

LOGGER = logging.getLogger(__name__)

class Article():
    """Create and enrich FoLiA documents.

    If you want to add more metadata to an article;

    .. code-block:: python

        Article_object.doc.metadata[<metada_datakey>] = <metadata>
        Article_object.doc.save()

    .. todo:: Document attributes

    Parameters
    ----------
    file_path : string, optional (default=None)
        Give the path that the document will be loaded/saved.

    folia_doc : object, optional (default=None)
        Give the FoLiA document object to be loaded.

    language : string, optional (default=None)
        ISO 639-1 code of the language.
    """
    def __init__(self, file_path=None, folia_doc=None, language=None):

        self.doc = None
        self.language = 'en'
        self.folia_xml_path = None
        self.folia_id = None

        if file_path:
            self.load_file(file_path=file_path)

    def load_file(self, file_path):
        """Reads and loads FoLiA document from file.

        Parameters
        ----------
        file_path : string
            Give the path that the document will be read.
        """
        self.folia_xml_path = os.path.abspath(file_path)

        if os.path.exists(self.folia_xml_path):
            self.doc = folia.Document(file=self.folia_xml_path)
            self.folia_id = self.doc.id

            LOGGER.debug("FoLiA XML file found and loaded.")
        else:
            LOGGER.warning("FoLiA XML file not found. Use create_doc() method " \
                           "to create one if not created yet.")
