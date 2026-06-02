import xml.etree.ElementTree as ET
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import extract_imaging_parameters


class NoMetadataError(Exception):
    pass


def load_metadata(metadata_path: str) -> dict:
    """Parse Bruker .env or .xml file and return imaging parameters.

    Returns dict with keys: x_um, y_um, z_um, width, height, n_planes (n_planes=None if not available).
    """
    tree = ET.parse(metadata_path)
    root = tree.getroot()

    microns = {}
    for state_value in root.findall('.//PVStateValue'):
        if state_value.get('key') == 'micronsPerPixel':
            for entry in state_value.findall('.//*[@index]'):
                idx = entry.get('index')
                if idx in ('XAxis', 'YAxis', 'ZAxis'):
                    microns[idx] = float(entry.get('value'))

    if 'XAxis' not in microns:
        raise NoMetadataError(f"micronsPerPixel not found in {metadata_path}")

    frame_rate, width, height = extract_imaging_parameters(metadata_path)

    return {
        'x_um': microns.get('XAxis', 1.0),
        'y_um': microns.get('YAxis', microns.get('XAxis', 1.0)),
        'z_um': microns.get('ZAxis', 1.0),
        'width': width,
        'height': height,
        'n_planes': None,
    }


def find_metadata_file(tiff_path: str) -> str | None:
    """Search same directory as tiff_path for a .env or .xml Bruker metadata file."""
    folder = os.path.dirname(tiff_path)
    for fname in os.listdir(folder):
        if fname.endswith('.env') or (fname.endswith('.xml') and not fname.endswith('.ome.tif')):
            return os.path.join(folder, fname)
    return None
