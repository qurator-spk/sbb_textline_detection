import json
import os
import tempfile

import click
import ocrd_models.ocrd_page
from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page_generateds import CoordsType, PageType
from ocrd_utils import (
    assert_file_grp_cardinality,
    getLogger,
    make_file_id,
    coordinates_for_segment,
    polygon_from_points, points_from_polygon,
)
import numpy as np
from shapely.geometry import Polygon, asPolygon
from shapely.ops import unary_union

from pkg_resources import resource_string

from qurator.sbb_textline_detector import textline_detector

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


    def process(self):
        log = getLogger('processor.OcrdSbbTextlineDetectorRecognize')
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

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = \
                self.workspace.image_from_page(
                    page, page_id,
                    feature_filter='cropped,binarized,grayscale_normalized'
                )

            with tempfile.TemporaryDirectory() as tmp_dirname:
                # Save the image
                image_file = tempfile.mkstemp(dir=tmp_dirname, suffix='.png')[1]
                page_image.save(image_file)

                # Segment the image
                model = self.parameter['model']
                x = textline_detector(image_file, tmp_dirname, file_id, model)
                x.run()

                # Read segmentation results
                tmp_filename = os.path.join(tmp_dirname, file_id) + '.xml'
                tmp_pcgts = ocrd_models.ocrd_page.parse(tmp_filename, silence=True)
                tmp_page = tmp_pcgts.get_Page()

            # Create a new PAGE file from the input file
            pcgts.set_pcGtsId(file_id)

            # Merge results → PAGE file

            # 1. Border
            if page.get_Border():
                log.warning("Removing existing page border")
            page.set_Border(None)
            # We need to translate the coordinates:
            text_border = adapt_coords(tmp_page.get_Border(), page, page_coords)
            if text_border is None:
                # intersection is empty (border outside of rotated original image)
                log.warning("new border would be empty, skipping")
            else:
                page.set_Border(text_border)

            # 2. ReadingOrder
            if page.get_ReadingOrder():
                log.warning("Removing existing regions' reading order")
            page.set_ReadingOrder(tmp_page.get_ReadingOrder())

            # 3. TextRegion
            # FIXME: what about table and image regions?
            if page.get_TextRegion():
                log.warning("Removing existing text regions")
            # We need to translate the coordinates:
            text_regions_new = []
            for text_region in tmp_page.get_TextRegion():
                text_region = adapt_coords(text_region, page, page_coords)
                if text_region is None:
                    # intersection is empty (polygon outside of above border)
                    log.warning("new text region polygon would be empty, skipping")
                    continue
                text_regions_new.append(text_region)
                text_lines_new = []
                for text_line in text_region.get_TextLine():
                    text_line = adapt_coords(text_line, text_region, page_coords)
                    if text_line is None:
                        # intersection is empty (polygon outside of region)
                        log.warning("new text line polygon would be empty, skipping")
                        continue
                    text_lines_new.append(text_line)
                text_region.set_TextLine(text_lines_new)
            page.set_TextRegion(text_regions_new)

            # Save metadata about this operation
            self.add_metadata(pcgts)

            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=page_id,
                mimetype='application/vnd.prima.page+xml',
                local_filename=os.path.join(self.output_file_grp, file_id) + '.xml',
                content=ocrd_models.ocrd_page.to_xml(pcgts)
            )


def adapt_coords(segment, parent, transform):
    points = segment.get_Coords().get_points()
    polygon = polygon_from_points(points)
    # polygon absolute coords (after transforming back from page coords, e.g. deskewing)
    polygon_new = coordinates_for_segment(polygon, None, transform)
    # intersection with parent polygon
    polygon_new = polygon_for_parent(polygon_new, parent)
    if polygon_new is None:
        return None
    points_new = points_from_polygon(polygon_new)
    segment.set_Coords(CoordsType(points=points_new))
    return segment

# from ocrd_tesserocr, to be integrated into core (somehow)...
def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.

    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0, 0], [0, parent.get_imageHeight()],
                               [parent.get_imageWidth(), parent.get_imageHeight()],
                               [parent.get_imageWidth(), 0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # check if clipping is necessary
    if childp.within(parentp):
        return polygon
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    # clip to parent
    interp = childp.intersection(parentp)
    if interp.is_empty or interp.area == 0.0:
        # this happens if Tesseract "finds" something
        # outside of the valid Border of a deskewed/cropped page
        # (empty corners created by masking); will be ignored
        return None
    if interp.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        # FIXME: construct concave hull / alpha shape
        interp = interp.convex_hull
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = asPolygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp.exterior.coords[:-1] # keep open

# from ocrd_tesserocr, to be integrated into core (somehow)...
def make_valid(polygon):
    for split in range(1, len(polygon.exterior.coords)-1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:]+polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon

if __name__ == '__main__':
    ocrd_sbb_textline_detector()
