"""Write VRM Animation (.vrma) files.

A .vrma is a glTF binary with the ``VRMC_vrm_animation`` extension: named
humanoid bone nodes whose ``rotation`` channels hold the pose, plus expression
nodes whose ``translation`` X component is the expression weight (that is what
three-vrm-animation reads).

The nodes carry no rest rotation, so the loader's rest-space conversion
collapses to identity and the tracks are simply the avatar's local bone
quaternions - exactly what the browser baked.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# expression nodes are ordered like the file, but the preset mapping is what
# matters - only VRM preset names are allowed here
EXPRESSION_PRESETS = [
    'happy', 'angry', 'sad', 'relaxed', 'surprised',
    'aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'blinkLeft', 'blinkRight',
]


def _align4(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def write_vrma(path: Path, data: dict) -> dict:
    """Convert the baked frame data into a .vrma file.

    ``data`` = {"fps": float, "frames": [{"t": sec,
                                          "bones": {name: [x,y,z,w]},
                                          "expr": {name: weight}}]}
    """
    frames = data.get('frames') or []
    if len(frames) < 2:
        raise ValueError('need at least two frames to build an animation')

    bone_names = []
    expr_names = []
    for frame in frames:
        for name in frame.get('bones', {}):
            if name not in bone_names:
                bone_names.append(name)
        for name in frame.get('expr', {}):
            if name in EXPRESSION_PRESETS and name not in expr_names:
                expr_names.append(name)

    times = [float(f['t']) for f in frames]
    # animation samplers expect a strictly increasing input
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('frame times must be strictly increasing')

    nodes = []
    human_bones = {}
    expressions = {}
    for name in bone_names:
        human_bones[name] = {'node': len(nodes)}
        node = {'name': name}
        if name == 'hips':
            # the loader checks the rest hips height is above the floor
            node['translation'] = [0.0, 1.0, 0.0]
        nodes.append(node)
    for name in expr_names:
        expressions[name] = {'node': len(nodes)}
        nodes.append({'name': f'VRMExpression_{name}'})

    buffer = bytearray()
    buffer_views = []
    accessors = []
    channels = []
    samplers = []

    # time accessor is shared by every sampler
    _align4(buffer)
    time_offset = len(buffer)
    time_raw = struct.pack('<%df' % len(times), *times)
    buffer.extend(time_raw)
    buffer_views.append({'buffer': 0, 'byteOffset': time_offset,
                         'byteLength': len(time_raw)})
    accessors.append({'bufferView': len(buffer_views) - 1,
                      'componentType': 5126, 'count': len(times),
                      'type': 'SCALAR',
                      'min': [times[0]], 'max': [times[-1]]})
    time_accessor = len(accessors) - 1

    def add_channel(node, path, values, comps):
        _align4(buffer)
        offset = len(buffer)
        flat = []
        for value in values:
            if comps == 1:
                flat.append(float(value))
            else:
                flat.extend(float(v) for v in value)
        raw = struct.pack('<%df' % len(flat), *flat)
        buffer.extend(raw)
        buffer_views.append({'buffer': 0, 'byteOffset': offset,
                             'byteLength': len(raw)})
        accessors.append({
            'bufferView': len(buffer_views) - 1,
            'componentType': 5126,
            'count': len(values),
            'type': {3: 'VEC3', 4: 'VEC4'}[comps],
        })
        samplers.append({'input': time_accessor,
                         'output': len(accessors) - 1,
                         'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1,
                         'target': {'node': node, 'path': path}})

    for name in bone_names:
        node = human_bones[name]['node']
        # bones missing a frame fall back to identity
        quats = [frame.get('bones', {}).get(name, [0.0, 0.0, 0.0, 1.0])
                 for frame in frames]
        add_channel(node, 'rotation', quats, 4)

    for name in expr_names:
        node = expressions[name]['node']
        # three-vrm-animation reads the X of a translation channel as weight
        values = [[float(frame.get('expr', {}).get(name, 0.0)), 0.0, 0.0]
                  for frame in frames]
        add_channel(node, 'translation', values, 3)

    gltf = {
        'asset': {'version': '2.0', 'generator': 'rtmlib-mocap'},
        'scene': 0,
        'scenes': [{'nodes': list(range(len(nodes)))}],
        'nodes': nodes,
        'animations': [{'channels': channels, 'samplers': samplers}],
        'bufferViews': buffer_views,
        'buffers': [{'byteLength': len(buffer)}],
        'accessors': accessors,
        'extensionsUsed': ['VRMC_vrm_animation'],
        'extensions': {
            'VRMC_vrm_animation': {
                'specVersion': '1.0',
                'humanoid': {'humanBones': human_bones},
                'expressions': {'preset': expressions},
            }
        },
    }

    _align4(buffer)
    json_chunk = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(json_chunk) % 4:
        json_chunk += b' '

    total = 12 + 8 + len(json_chunk) + 8 + len(buffer)
    blob = bytearray()
    blob += struct.pack('<III', 0x46546C67, 2, total)
    blob += struct.pack('<II', len(json_chunk), 0x4E4F534A) + json_chunk
    blob += struct.pack('<II', len(buffer), 0x004E4942) + buffer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {'bytes': len(blob), 'bones': len(bone_names),
            'expressions': len(expr_names), 'frames': len(frames)}
