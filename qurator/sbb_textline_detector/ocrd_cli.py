import json
import os

import click
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_modelfactory import page_from_file
from ocrd_models import OcrdFile
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE
from pkg_resources import resource_string

from qurator.sbb_textline_detector import textlineerkenner

log = getLogger('processor.OcrdSbbTextlineDetectorRecognize')

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))


@click.command()
@ocrd_cli_options
def ocrd_sbb_textline_detector(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdSbbTextlineDetectorRecognize, *args, **kwargs)


class OcrdSbbTextlineDetectorRecognize(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd_sbb_textline_detector']
        super(OcrdSbbTextlineDetectorRecognize, self).__init__(*args, **kwargs)

    def _make_file_id(self, input_file, input_file_grp, n):
        file_id = input_file.ID.replace(input_file_grp, self.output_file_grp)
        if file_id == input_file.ID:
            file_id = concat_padded(self.output_file_grp, n)
        return file_id

    def _resolve_image_file(self, input_file: OcrdFile) -> str:
        if input_file.mimetype == MIMETYPE_PAGE:
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            image_file = page.imageFilename
        else:
            image_file = input_file.local_filename
        return image_file

    def process(self):
        for n, page_id in enumerate(self.workspace.mets.physical_pages):
            input_file = self.workspace.mets.find_files(fileGrp=self.input_file_grp, pageId=page_id)[0]
            log.info("INPUT FILE %i / %s", n, input_file)

            file_id = self._make_file_id(input_file, self.output_file_grp, n)
            image_file = self._resolve_image_file(input_file)

            # Process the files
            try:
                os.mkdir(self.output_file_grp)
            except FileExistsError:
                pass

            model = self.parameter['model']
            x = textlineerkenner(image_file, self.output_file_grp, file_id, model)
            x.run()

            self.workspace.add_file(
                 ID=file_id,
                 file_grp=self.output_file_grp,
                 pageId=page_id,
                 mimetype='application/vnd.prima.page+xml',
                 local_filename=os.path.join(self.output_file_grp, file_id) + '.xml')


if __name__ == '__main__':
    ocrd_sbb_textline_detector()
