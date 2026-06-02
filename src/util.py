import xml.etree.ElementTree as ET


def extract_imaging_parameters(metadata_file):
    """Extract frame rate and frame dimensions from XML metadata file."""
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    frame_rate = None
    frame_width = None
    frame_height = None

    for state_value in root.findall('.//PVStateValue'):
        key = state_value.get('key')
        if key == 'framerate':
            frame_rate = float(state_value.get('value'))
        elif key == 'pixelsPerLine':
            frame_width = int(state_value.get('value'))
        elif key == 'linesPerFrame':
            frame_height = int(state_value.get('value'))

    print(f"Frame size: {frame_width} x {frame_height} pixels")
    print(f"Frame rate: {frame_rate:.2f} Hz")

    return frame_rate, frame_width, frame_height


def read_microns_per_pixel(metadata_file):
    """Read micronsPerPixel (XAxis) from PrairieView XML metadata file."""
    tree = ET.parse(metadata_file)
    root = tree.getroot()

    for state_value in root.findall('.//PVStateValue'):
        if state_value.get('key') == 'micronsPerPixel':
            for entry in state_value.findall('.//*[@index]'):
                if entry.get('index') == 'XAxis':
                    return float(entry.get('value'))

    raise ValueError(f"micronsPerPixel not found in {metadata_file}")
