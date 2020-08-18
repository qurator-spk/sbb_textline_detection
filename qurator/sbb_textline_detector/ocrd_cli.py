import json
import os
import tempfile

import click
import ocrd_models.ocrd_page
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_models.ocrd_page_generateds import MetadataItemType, LabelsType, LabelType
from ocrd_utils import (
    assert_file_grp_cardinality,
    getLogger,
    make_file_id,
    MIMETYPE_PAGE
)

from pkg_resources import resource_string

from qurator.sbb_textline_detector import textline_detector

log = getLogger('processor.OcrdSbbTextlineDetectorRecognize')

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))


@click.command()
@ocrd_cli_options
def ocrd_sbb_textline_detector(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdSbbTextlineDetectorRecognize, *args, **kwargs)


TOOL = 'ocrd-sbb-textline-detector'

class OcrdSbbTextlineDetectorRecognize(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdSbbTextlineDetectorRecognize, self).__init__(*args, **kwargs)

    def _resolve_image_file(self, input_file: OcrdFile) -> str:
        if input_file.mimetype == MIMETYPE_PAGE:
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            image_file = page.imageFilename
        else:
            image_file = input_file.local_filename
        return image_file

    def process(self):
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            log.info("INPUT FILE %i / %s", n, input_file)

            file_id = make_file_id(input_file, self.output_file_grp)

            # Process the files
            try:
                os.mkdir(self.output_file_grp)
            except FileExistsError:
                pass

            with tempfile.TemporaryDirectory() as tmp_dirname:
                # Segment the image
                image_file = self._resolve_image_file(input_file)
                model = self.parameter['model']
                x = textline_detector(image_file, tmp_dirname, file_id, model)
                x.run()

                # Read segmentation results
                tmp_filename = os.path.join(tmp_dirname, file_id) + '.xml'
                tmp_pcgts = ocrd_models.ocrd_page.parse(tmp_filename, silence=True)
                tmp_page = tmp_pcgts.get_Page()

            # Create a new PAGE file from the input file
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pcgts.set_pcGtsId(file_id)
            page = pcgts.get_Page()

            # Merge results → PAGE file
            if page.get_Border():
                log.warning("Page already contained a border")
            page.set_Border(tmp_page.get_Border())
            if page.get_ReadingOrder():
                log.warning("Page already contained a reading order")
            page.set_ReadingOrder(tmp_page.get_ReadingOrder())
            if page.get_TextRegion():
                log.warning("Page already contained text regions")
            page.set_TextRegion(tmp_page.get_TextRegion())

            # Save metadata about this operation
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name, value=self.parameter[name])
                                            for name in self.parameter.keys()])]))

            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.output_file_grp,
                 pageId=page_id,
                 mimetype='application/vnd.prima.page+xml',
                 local_filename=os.path.join(self.output_file_grp, file_id) + '.xml',
                 content=ocrd_models.ocrd_page.to_xml(pcgts)
            )


if __name__ == '__main__':
    ocrd_sbb_textline_detector()
