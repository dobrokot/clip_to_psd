'''

This script outputs all PSD data sections as readable text or PNG files.
It decodes some binary data properties to readable text and descends into sub-containers of properties, but this support far from complete. Adobe binary "Object Descriptor" and output of TextEngine code are supported.

Main purpose of this script is reverse engineering of PSD files to assist creation of clip_to_psd.py.
Also could be used to explore internals of PSD files or export image data, but it's secondary purpose and script usability is not optimized well for these tasks.

Global variables below are used as config to skip some huge (xmp, thumb) or frequently changed (shmd with timestamp) properties. Maybe it's better as command line parameters. But this is supplementary internal tool and python data structures are more readable than command line strings.


used as:
python psd_parse input.psd directory_for_exported_layers_data

'''

import sys
import struct
import re
import io
import os
from collections import namedtuple
from PIL import Image

decode_and_save_to_png = True # affects speed greatly. If you don't need layer images in png, only meta-data, set this to False.
print_channel_compression_sizes = False # useful for diff tool to compare edited .PSDs and avoid noise in diff

skip_global_properties_print = {
    1060,  # xmp
    1036,  # thumb
    1058,  # exif
}

skip_layer_tags_print = {
    'shmd'
}

print_readable = True # skip or include text interpretation of some binary data
print_full_binary = True # could be used as protecton multi-kilobyte output for meangless huge secions
print_zero_byte_as_dot = False # could be a bit more readable for long sequends of \x00\x00..., but makes output ambigous

zero_replace = b'.' if print_zero_byte_as_dot else b'\0'

def get_unicode_string(f):
    str_len = get_int(f)
    data = read_checked(f, str_len*2)
    # sometimes '\0' present in the end, often doesn't. Keep it here to make difference in text output
    return data.decode('UTF-16BE') #.rstrip('\0')

def parse_object_descriptor(f, depth = 0):
    offset = [0]
    #debug_read_file(f)
    def prn(*x, **y):
        print(' ' * (depth * 4), *x, **y)

    name = get_unicode_string(f)
    prn(f"name for class id: {repr(name)}")

    def read_key_name():
        n = get_int(f)
        mark0 = b'~' # make difference between 4 and 0 to reflect any binary difference in text output
        if n == 0:
            mark0 = b''
            n = 4
        return mark0 + read_checked(f, n)

    def parse_item(type_name, depth):
        # called in context with depth indentation already made, so 'print' used instead prn
        if type_name == b'enum':
            enum_type = read_key_name()
            enum_value = read_key_name()
            print(f"{enum_type=} {enum_value=}")
        elif type_name == b'TEXT':
            text = get_unicode_string(f)
            #print(f"'{text}'")
            print(f"{repr(text)}")
        elif type_name in (b'Objc', b'GlbO'):
            print()
            parse_object_descriptor(f, depth + 1)
        elif type_name == b'UntF':
            unit_name=read_checked(f, 4)
            value = get_double(f)
            print(f"{unit_name=} {value=}")
        elif type_name == b'long':
            value = get_int(f)
            print(f"{value=}")
        elif type_name == b'doub':
            value = get_double(f)
            print(f"{value=}")
        elif type_name == b'bool':
            value = bool(get_int(f, 1))
            print(f"{value=}")
        elif type_name in (b'type', b'GlbC', b'Clss'):
            #untested
            item_name = get_unicode_string(f)
            item_class_id = read_key_name()
            print(f"{item_name=} {item_class_id=}")
        elif type_name in (b'tdta', b'alis', b'Pth '):
            n = get_int(f)
            data = read_checked(f, n)
            if key_name == b'~EngineData':
                print()
                #import pprint
                #pprint.pprint(data)
                print(replace_embedded_unicode_to_text(data.decode('latin-1')))
                print()
            else:
                print(f"{data=}")
        elif type_name in (b'VlLs', b'obj '):
            # 'obj ' untested
            list_count = get_int(f)
            print(f"{list_count=}")
            for _j in range(list_count):
                list_item_type_name = read_checked(f, 4)
                prn(f"{list_item_type_name=}", end = ' ')
                parse_item(list_item_type_name, depth + 1)
        else:
            prn(f"not implemented reading for this field, '{type_name=}'")
            raise StopIteration

    class_id = read_key_name()
    prn(f'{class_id=}')
    item_count = get_int(f)
    prn(f'{item_count=}')

    for _i in range(item_count):
        key_name = read_key_name()
        type_name = read_checked(f, 4)
        prn(f'{key_name=} {type_name=}', end = ' ')
        parse_item(type_name, depth)

def parse_object_descriptor_with_version(f):
    descriptor_version = get_int(f)
    print(f'{descriptor_version=}')
    try:
        parse_object_descriptor(f)
    except StopIteration:
        print("can't parse full, truncated output")


def print_entries_array_compact_defaults(entries):
    # lvls has about 60 identical default non-used values, print them in compact form.
    DEFAULT_LEVL_ENTRY = [0, 255, 0, 255, 100]
    if not entries:
        print('[empty array]')
        return

    last_non_default = -1
    for i in reversed(range(len(entries))):
        if entries[i] != DEFAULT_LEVL_ENTRY:
            last_non_default = i
            break
    entries_to_print = entries[0:last_non_default + 2] # print one default after last non-default to show what it looks like

    for entry in entries_to_print:
        default_str = ' (default)' if (entry == DEFAULT_LEVL_ENTRY) else ''
        input_floor, input_ceil, output_floor, output_ceil, inversed_gamma = entry
        print(f'{input_floor=} {input_ceil=} {output_floor=} {output_ceil=} {inversed_gamma=}' + default_str)

    non_printed_count = len(entries) - len(entries_to_print)
    if non_printed_count > 0:
        print(f"skipping {non_printed_count} other default entries")

def parse_levels_adjustment(f):
    lvls_version = get_int(f, 2)
    print(f'{lvls_version=}')
    if lvls_version != 2:
        print(f'unusual value of {lvls_version=}, expected 2')

    print_entries_array_compact_defaults([[get_int(f, 2) for _ in range(5)] for _i in range(29)])

    cur = f.tell()
    f.seek(0, 2)
    end = f.tell()
    if cur == end:
        return
    f.seek(cur, 0)

    tag_extra = f.read(4)
    if tag_extra != b'Lvls':
        print("expected extra 'Lvls' tag inside'lvls', but found something other.")
        return

    lvls_ext_version = get_int(f, 2)
    print(f'{lvls_ext_version=}')

    lvls_ext_count = get_int(f, 2)
    lvls_ext_entries_count = lvls_ext_count - 29
    print(f'{lvls_ext_count=}, lvls_ext_count-29 = {lvls_ext_entries_count} entries count')
    print_entries_array_compact_defaults([[get_int(f, 2) for _ in range(5)] for _i in range(lvls_ext_entries_count)])
    extra_bytes = f.read()
    if (extra_bytes):
        print("extra bytes:", extra_bytes.hex(' '))

def parse_curve_adjustment(f):
    is_mapping = get_int(f, 1)
    print(f"{is_mapping=}")
    version = get_int(f, 2)
    print(f"{version=}")

    channel_bit_set = get_int(f, 4)
    print(f"channel_bit_set: {bin(channel_bit_set)}")
    count_channels = bin(channel_bit_set).count('1')

    def read_points():
        if is_mapping:
            print('points mapping:', [int(x) for x in f.read(256)])
        else:
            count_of_points = get_int(f, 2)
            print(f"{count_of_points=}")
            for _ in range(count_of_points):
                x, y = get_int(f, 2), get_int(f, 2)
                print(f"Point: ({x}, {y})")

    for _ in range(count_channels):
        read_points()

    crv = f.read(4)
    if crv != b'Crv ':
        extra_data = crv + f.read()
        if extra_data:
            print('extra data:', extra_data.hex(' '))

    crv_version = get_int(f, 2)
    print(f"Crv version: {crv_version}")

    count_channels_in_extra = get_int(f, 4)
    print(f"{count_channels_in_extra=}")

    for _ in range(count_channels_in_extra):
        channel_id = get_int(f, 2)
        print(f"{channel_id=}")
        read_points()

    extra_data = f.read()
    if extra_data:
        print('extra data:', extra_data.hex(' '))

def parse_hue_saturation(f):
    f.seek(0, 2)
    eof = f.tell()
    f.seek(0, 0)
    version = get_int(f, 2)
    print(f"{version=} # Version")

    # Use settings for hue-adjustment or colorization
    hue_adjustment = get_int(f, 1)
    print(f"{hue_adjustment=} # meaning: 0: Use settings for hue-adjustment; 1: Use settings for colorization.\n")

    # Padding byte
    padding = get_int(f, 1)
    print(f"{padding=}")

    # Colorization
    colorization_hue = get_int(f, 2, True)
    colorization_saturation = get_int(f, 2, True)
    colorization_lightness = get_int(f, 2, True)
    print(f"{colorization_hue=}")
    print(f"{colorization_saturation=}")
    print(f"{colorization_lightness=}")

    # Master hue, saturation and lightness values
    master_hue = get_int(f, 2, True)
    master_saturation = get_int(f, 2, True)
    master_lightness = get_int(f, 2, True)
    print(f"{master_hue=}")
    print(f"{master_saturation=}")
    print(f"{master_lightness=}")

    while f.tell() != eof:
        range_values = [get_int(f, 2, True) for _ in range(4)]
        settings_values = [get_int(f, 2, True) for _ in range(3)]
        print(f"{range_values=} {settings_values=}")


def parse_tysh_txt_tag(f):
    version = get_int(f, 2)
    if version != 1:
        print("expected version 1 for 'TySh' text section")
        return

    print('matrix:')
    for _i in range(3):
        print(get_double(f), get_double(f))

    version2 = get_int(f, 2)
    if version2 != 50:
        if version2 == 6:
            print('old photoshop 5.0 TySh text section version')
        else:
            print('unknown TySh text section version', version2)
        return

    parse_object_descriptor_with_version(f)
    warp_version = get_int(f, 2)
    print(f'{warp_version=}')
    print('Warp data:')
    parse_object_descriptor_with_version(f)
    left, top, right, bottom = [get_int(f) for _ in range(4)]
    print(f'{left=} {top=} {right=} {bottom=}')
    unread_data = f.read()
    print('tysh read section size', f.tell())
    if unread_data:
        print('unread data:', unread_data)


# much faster to create all possible bytearrays at start then create them in the inner loop of decode_rle. Takes about 0.05sec at start and 5mb of RAM
table_of_all_bytearrays = [[bytearray([i]*l) for i in range(0, 256)] for l in range(128+1)]

def decode_rle(data, start, end, out_size):
    result = bytearray(out_size)
    s = start
    d = 0
    while s < end:
        hdr = data[s] 
        s += 1
        if hdr <= 127:
            # copy sequence from source
            l = hdr + 1
            result[d:d + l] = data[s:s + l]
            s += l
            d += l
        elif hdr != 128: # value 128: just skip the byte, according to documentation
            # repeat 1 byte many times. Preallocated table is used to avoid object creation in loop.
            l = 257 - hdr
            result[d:d + l] = table_of_all_bytearrays[l][data[s]]
            s += 1
            d += l

    # python specific hack: check bounds error after all array writes. It works because a[x:y] doesn't produce error
    if s > end:
        raise ValueError('invalid RLE compressed data, source read overflow')
    if d > out_size:
        raise ValueError('invalid RLE compressed data, destination write overflow')

    if d != out_size:
        raise ValueError(f'expected {out_size} bytes for scanline, but get only {d} bytes')

    return result


def read_checked(f, size):
    assert size >= 0
    data = f.read(size)
    assert (len(data) == size), (len(data), size)
    return data

def get_int(f, size = 4, signed = False): return int.from_bytes(read_checked(f, size), 'big', signed=signed)
def get_double(f): return struct.unpack('>d', read_checked(f, 8))[0]
def get_int_psb(f, psd_version): return int.from_bytes(read_checked(f, 4 * psd_version), 'big')

def read_bytestr_padded(f, padding):
    length = get_int(f, 1)
    result = read_checked(f, length)
    read_checked(f, -(length+1) % padding)
    return result

def get_struct(f, fmt):
    size = struct.calcsize(fmt)
    data = read_checked(f, size)
    assert size == len(data)
    return struct.unpack(">" + fmt, data)

def get_struct_1(f, fmt):
    result = get_struct(f, fmt)
    assert len(result) == 1
    return result

MaskFlags = namedtuple("MaskFlags", [
    "pos_relative_to_layer",
    "mask_disabled",
    "mask_invert",
    "mask_from_render",
    "has_parameters",
    "mask_other_flags",
])

def read_mask_flag(f):
    flags = get_int(f, 1)
    print(f"mask flags {flags}")
    return MaskFlags(
        bool(flags & 1),
        bool(flags & 2),
        bool(flags & 4),
        bool(flags & 8),
        bool(flags & 16),
        bool(flags & 0b11100000),
    )

MaskParameters = namedtuple("MaskParameters", [ "user_mask_density", "user_mask_feather", "vector_mask_density", "vector_mask_feather" ])

#pylint: disable=attribute-defined-outside-init
def read_mask(f):
    section_size = get_int(f)

    print(f"mask {section_size=}")
    if section_size == 0:
        return None

    mask_section_data = f.read(section_size)
    f = io.BytesIO(mask_section_data)

    #pylint: disable=too-many-instance-attributes
    class Mask:
        pass

    m = Mask()
    m.top, m.left, m.bottom, m.right = get_struct(f, "4i")
    print(f"mask {m.top=}, {m.left=}, {m.bottom=}, {m.right=}")
    m.default_color_int8 = get_int(f, 1)
    print(f"mask {m.default_color_int8=}")
    m.flags = read_mask_flag(f)
    print(f"mask {m.flags=}")

    m.has_real = False
    m.real_flags, m.real_default_color_int8 = None, None
    m.real_top, m.real_left, m.real_bottom, m.real_right = None, None, None, None
    if section_size >= 36:
        m.has_real = True
        m.real_flags = read_mask_flag(f)
        m.real_default_color_int8 = get_int(f, 1)
        m.real_top, m.real_left, m.real_bottom, m.real_right = get_struct(f, "4i")

    m.parameters = (None, None, None, None)
    if m.flags.has_parameters:
        p = get_int(f, 1)
        m.parameters = MaskParameters(
            get_int(f, 1) if (p & 1) else None,
            get_struct_1(f, 'd') if (p & 2) else None,
            get_int(f, 1) if (p & 4) else None,
            get_struct_1(f, 'd') if (p & 8) else None,
        )

    #f.seek(section_start + section_size, 0)
    return m


def debug_read_file(f):
    f_read = f.read
    def my_read(k = -1):
        result = f_read(k)
        if len(result) <= 32:
            print(f'f.read({k}):', result.hex(' '), repr(result))
        else:
            print(
                f'f.read({k}):',
                result[0:16].hex(' ') + (' ... ') + result[-16:].hex(' '),
                repr(result[0:16]) + (' ... ') + repr(result[-16:])
            )
        return result
    f.read = my_read


tags_for_8_byte_length_size = set(b'LMsk Lr16 Lr32 Layr Mt16 Mt32 Mtrn Alph FMsk lnk2 FEid FXid PxSD'.split())
def parse_psd_tags(extra_data_stream, padding, psd_version):
    extra_data_list = []
    while sig := extra_data_stream.read(4):
        assert sig in (b'8BIM', b'8B64'), repr(sig)
        key = read_checked(extra_data_stream, 4)
        long_size = (key in tags_for_8_byte_length_size) and psd_version == 2
        size_of_size = (8 if long_size else 4)
        size = get_int(extra_data_stream, size_of_size)
        extra_data_entry = read_checked(extra_data_stream, size)
        assert len(extra_data_entry) == size
        read_checked(extra_data_stream, -size % padding)
        extra_data_list.append((key, extra_data_entry))
    return extra_data_list


def read_layer(f, psd_version):
    class Layer:
        pass
    l = Layer()

    l.top, l.left, l.bottom, l.right = get_struct(f, "4i")
    print('rect:', l.top, l.left, l.bottom, l.right)
    assert (abs(l.top) + abs(l.left)  + abs(l.bottom) + abs(l.right)) < 10*1000*1000
    channel_count = get_int(f, 2)
    l.channels = []
    assert channel_count < 20, channel_count
    for _i_channel in range(channel_count):
        (channel_id,) = get_struct(f, "h")
        #assert channel_id == i_channel
        channel_data_size = get_int_psb(f, psd_version)
        l.channels.append((channel_id, channel_data_size))
    if print_channel_compression_sizes:
        print('channels:', l.channels)
    else:
        print('channels:', [x[0] for x in l.channels])
    assert b'8BIM' == get_struct(f, "4s")[0]
    l.blend_mode = get_struct(f, "4s")[0]
    blend_mode_description = psd_blend_description.get(l.blend_mode, "???")
    print('blend:', l.blend_mode, f'({blend_mode_description})')
    l.opacity_int8 = get_int(f, 1)
    print(f'{l.opacity_int8=}')
    l.clipping_bool = bool(get_int(f, 1))
    print(f'{l.clipping_bool=}')
    flags = get_int(f, 1)
    l.locked_alpha = bool(flags & 1)
    l.visible = not bool(flags & 2)
    _pixel_data_irrelevant_to_appearance = bool((flags & 8) and (flags & 16))
    print(f'flags: {flags:x}, {l.locked_alpha=} {l.visible=}')
    get_int(f, 1) #padding

    k = get_int(f)
    extra_data = read_checked(f, k)
    print('layer extra data size:', k)

    #print(extra_data.hex(' '))
    #print(extra_data.replace(b'\0', b'.'))

    with io.BytesIO(extra_data) as extra_data_stream:
        l.mask = read_mask(extra_data_stream)
        k = get_int(extra_data_stream) # skip "Layer blending ranges"
        read_checked(extra_data_stream, k)

        l.name = read_bytestr_padded(extra_data_stream, 4).decode('cp1251', 'replace') # actually name is taken from luni unicode section

        extra_data_list = parse_psd_tags(extra_data_stream, 1, psd_version)


    if l.mask:
        m = l.mask
        mask_str = f'mask: {m.left=} {m.top=} {m.right=} {m.bottom=} {m.default_color_int8=} {m.flags=}'
        if m.has_real:
            mask_str += f'{m.real_left=} {m.real_top=} {m.real_right=} {m.real_default_color_int8=} {m.real_flags=}'
        if any(x != None for x in m.parameters):
            mask_str += f'{m.parameters}'
        print(mask_str)

    print('Layer name:', l.name)
    for key, extra_data_entry in extra_data_list:
        if key.decode('ascii') in skip_layer_tags_print:
            continue

        limit = None if print_full_binary else 1000

        print(key, extra_data_entry[0:limit].hex(' '), extra_data_entry[0:limit].replace(b'\0', zero_replace))

        descriptor_based_data_headers = (b'SoCo', b'CgEd', b'PtFl', b'GdFl', b'blwh', b'CgEd', b'vibA', b'pths', b'anFX', b'vstk', b'PxSc', b'cinf', b'artb', b'artd', b'abdd')
        if key == b'TySh' and print_readable:
            parse_tysh_txt_tag(io.BytesIO(extra_data_entry))
        if key == b'lfx2' and print_readable:
            f = io.BytesIO(extra_data_entry)
            effects_version = f.read(4)
            print(f'{effects_version=}')
            parse_object_descriptor_with_version(f)
        if key in descriptor_based_data_headers and print_readable:
            parse_object_descriptor_with_version(io.BytesIO(extra_data_entry))
        if key == b'levl':
            parse_levels_adjustment(io.BytesIO(extra_data_entry))
        if key == b'curv':
            parse_curve_adjustment(io.BytesIO(extra_data_entry))
        if key == b'hue2':
            parse_hue_saturation(io.BytesIO(extra_data_entry))

    return l

def decode_image_data(compression_type, channel_data, psd_version, width, height):
    if compression_type == 0:
        data_decoded = channel_data
        #width2 = width + width%2 # test this
        width2 = width
    elif compression_type == 1:
        k = 2 * psd_version
        byte_lens_binary = channel_data[0:height * k]
        byte_lens = [int.from_bytes(byte_lens_binary[k*i:k*i+k], 'big') for i in range(height)]
        #print(byte_lens)
        start = len(byte_lens_binary)
        decoded_lines = []
        for byte_len in byte_lens:
            end = start+byte_len
            #print('start-end', start,end)
            scanline_decoded = decode_rle(channel_data, start, end, width)
            start = end
            assert len(scanline_decoded) == width
            #scanline_decoded = b'x' * width
            decoded_lines.append(scanline_decoded)

        data_decoded = b''.join(decoded_lines)
        width2 = width
    else:
        assert False, (compression_type)

    assert len(data_decoded) == width2 * height, (len(data_decoded), width2, height, width2*height, width2*height - len(data_decoded))
    return data_decoded, width2

def decode_channel(channel_type, compression_type, channel_data, psd_version, l):
    if channel_type >= -1:
        rect = l
    elif channel_type == -2:
        assert l.mask
        rect = l.mask
    else:
        assert False, channel_type
    width, height = (rect.right - rect.left, rect.bottom - rect.top)
    if width == 0 or height == 0:
        return None, width, width, height

    data_decoded, width2 = decode_image_data(compression_type, channel_data, psd_version, width, height)
    return data_decoded, width, width2, height


def save_layer_to_file(channel_type_to_img, l, used_names, out_dir):
    output_imgs = []

    letter_to_channel_type = { 'A':-1, 'R':0,'G':1, 'B':2 } # -2 - mask, -3 - extra real mask

    for channel_type, img in channel_type_to_img.items():
        if channel_type <= -2:
            output_imgs.append((img, f'_mask{2-channel_type}'))

    if all((letter_to_channel_type[c] in channel_type_to_img) for c in 'RGBA'):
        img = Image.merge('RGBA', tuple(channel_type_to_img[letter_to_channel_type[c]] for c in 'RGBA'))
        output_imgs.append((img, ''))
    elif all((letter_to_channel_type[c] in channel_type_to_img) for c in 'RGB'):
        img = Image.merge('RGB', tuple(channel_type_to_img[letter_to_channel_type[c]] for c in 'RGB'))
        output_imgs.append((img, ''))
    else:
        for channel_type, img in channel_type_to_img.items():
            if channel_type >= -1:
                output_imgs.append((img, f'channel{1+channel_type}'))


    for img, name_suffix in output_imgs:
        layer_name_sanitized = re.sub(r'''[\0-\x1f'"/\\:*?<>| ]''', '_', l.name)
        layer_name_sanitized = layer_name_sanitized[0:100]
        i = 0
        result_name = layer_name_sanitized + name_suffix
        while result_name in used_names:
            i += 1
            result_name = layer_name_sanitized + f'({i})'
        used_names.add(result_name)
        out_fn = 'out_layer_' + result_name + '.png'
        img.save(os.path.join(out_dir, out_fn))


def parse_layers_pixels_and_save(f, layers, psd_version, out_dir):
    used_names = set()
    for l in layers:
        channel_type_to_img = {}
        #channel_type_to_letter = { -1: 'A', 0: 'R', 1: 'G', 2: 'B' } # -2 - mask, -3 - extra real mask
        if print_channel_compression_sizes:
            print()
            print('Layer name:', l.name, 'channels:', l.channels)
            print((l.right - l.left, l.bottom - l.top))


        for channel_type, channel_data_size in l.channels:
            compression_type = get_int(f, 2)

            channel_data = read_checked(f, channel_data_size - 2)
            if print_channel_compression_sizes:
                print(f'{channel_type=}, {compression_type=}')
                print(channel_type, channel_data_size)
            if decode_and_save_to_png:
                data_decoded, _width, width2, height = decode_channel(channel_type, compression_type, channel_data, psd_version, l)
                if data_decoded:
                    channel_type_to_img[channel_type] = Image.frombuffer("L", (width2, height), data_decoded, 'raw')

        if decode_and_save_to_png:
            save_layer_to_file(channel_type_to_img, l, used_names, out_dir)


def replace_embedded_unicode_to_text(text):
    # psd text engine language stored as ascii with embedded 2-bytes unicode, starting with '(\fe\ff' and
    # ending with ')'. To tell difference from unicode bytes containing byte ')' from ')' as end of embedding,
    # internal ')' and few other characters ('(', '\')  are escaped by slash. Unicode character in binary can produce
    # any bytes, including zero and EOL, so re.DOTALL is used.

    def replace(m):
        new_text = m.group(1).replace('\\', '').encode('latin-1').decode('UTF-16BE')
        return repr(new_text)

    return re.sub('[(]\u00FE\u00FF((@@)*)[)]'.replace('@', r'([\\].|[^)])'), replace, text, count = 0, flags = re.DOTALL)

def parse_layers(f, psd_version, out_dir):
    _layers_full_section_start = f.tell()
    layers_full_section_size = get_int_psb(f, psd_version)
    if layers_full_section_size == 0:
        print('no layers, layer_count=0, only composite image')
        return []
    print(f'{layers_full_section_size=}')

    replace_file_to_array = True
    if replace_file_to_array:
        f_data = read_checked(f, layers_full_section_size)
        f = io.BytesIO(f_data)
        layers_full_section_start = 0

    _layers_full_section_data_start = f.tell()
    _layers_info_subsection_start = f.tell()
    layers_info_subsection_size = get_int_psb(f, psd_version)
    print(f'{layers_info_subsection_size=}')
    if layers_info_subsection_size == 0:
        print('no layers, layer_count=0, only composite image(2)')
        return []

    layers_info_subsection_data = read_checked(f, layers_info_subsection_size)
    rest_of_full_layers_section_size = layers_full_section_size - 4*psd_version - layers_info_subsection_size
    rest_of_full_layers_section = read_checked(f, rest_of_full_layers_section_size)
    assert rest_of_full_layers_section_size == len(rest_of_full_layers_section)
    if replace_file_to_array:
        assert not f.read(1)

    with io.BytesIO(layers_info_subsection_data) as f2:
        layer_count = get_int(f2, 2)
        if layer_count >= 2**15:
            layer_count -= 2**16
        layer_count_negative = layer_count < 0
        layer_count = abs(layer_count)
        print(f'{layer_count=}, {layer_count_negative=}')
        assert layer_count < 10000, layer_count
        layers = []
        for _i_layer in range(layer_count):
            print()
            layers.append(read_layer(f2, psd_version))

        parse_layers_pixels_and_save(f2, layers, psd_version, out_dir)

    print()
    print('-------------------------------------')
    limit = None if print_full_binary else 100
    print(rest_of_full_layers_section[0:limit])
    #print(rest_of_full_layers_section)
    print(repr(rest_of_full_layers_section[0:limit].replace(b'\0', zero_replace)))

    print('size of data after layers info section', len(rest_of_full_layers_section))
    if rest_of_full_layers_section:
        with io.BytesIO(rest_of_full_layers_section) as global_tags_stream:
            global_mask_size = get_int(global_tags_stream, 4)
            print("global mask size:", global_mask_size)
            #debug_read_file(global_tags_stream)
            read_checked(global_tags_stream, global_mask_size)

            global_tags_list = parse_psd_tags(global_tags_stream, 4, psd_version)
            print("global tags:")
            for key, extra_data_entry in global_tags_list:
                #print(key, extra_data_entry.hex(' '), extra_data_entry.replace(b'\0', b'.'))
                if key == b'Txt2' and print_readable:
                    print(key)
                    print(replace_embedded_unicode_to_text(extra_data_entry.decode("latin-1")).rstrip('\0').replace('>>', '>>\n').replace('<<', '\n<<'))
                    #print(replace_embedded_unicode_to_text(extra_data_entry.decode("latin-1")))
                else:
                    limit = None if print_full_binary else 1000
                    print(key, extra_data_entry[0:limit].hex(' '), extra_data_entry[0:limit].replace(b'\0', zero_replace))


def parse_global_image(f, psd_version, out_dir, channels, width, height):
    global_image_compression_type = get_int(f, 2)
    global_image_data = f.read()

    if print_channel_compression_sizes:
        print(f'{global_image_compression_type=}, {len(global_image_data)=}')

    if not decode_and_save_to_png:
        return

    data_decoded, width2 = decode_image_data(global_image_compression_type, global_image_data, psd_version, width, height*channels)
    assert channels in (1, 3, 4), channels
    print(f'global image data size: {len(data_decoded)}')
    plane_size = width*height
    image_planes = [Image.frombuffer("L", (width2, height), data_decoded[i*plane_size:(i+1)*plane_size], 'raw') for i in range(channels)]
    image_mode = [None, "L", None, "RGB", "RGBA"][channels]
    assert image_mode
    global_image_img = Image.merge(image_mode, image_planes)
    global_image_img.save(os.path.join(out_dir, 'full_image.png'))


def parse_image_resource_slices_v6(f):
    # 'slices' is pretty rarely used feature of Photoshop, 
    # it was used to test parser of some non-trivial Object Descriptor binary
    cur = f.tell()
    end = f.seek(0, 2)
    f.seek(cur, 0)

    bbox_top_left_bottom_right = [get_int(f) for _ in range(4)]
    name = get_unicode_string(f)
    count = get_int(f)
    print(f'{bbox_top_left_bottom_right=}, {name=}, {count=}')
    for _ in range(count):
        #debug_read_file(f)
        slice_id = get_int(f)
        print(f'{slice_id=}')
        slice_group_id = get_int(f)
        print(f'{slice_group_id=}')
        slice_origin = get_int(f)
        print(f'{slice_origin=}')
        slice_associated_layer_id = None
        if slice_origin == 1:
            # untested
            slice_associated_layer_id = get_int(f)
        print(f'{slice_associated_layer_id=}')
        slice_name = get_unicode_string(f)
        print(f'{slice_name=}')
        slice_type = get_int(f)
        print(f'{slice_type=}')
        slice_left_top_right_bottom_box = [get_int(f) for _ in range(4)]
        print(f'{slice_left_top_right_bottom_box=}')
        slice_url = get_unicode_string(f)
        print(f'{slice_url=}')
        slice_target = get_unicode_string(f)
        print(f'{slice_target=}')
        slice_message = get_unicode_string(f)
        print(f'{slice_message=}')
        slice_alt_tag = get_unicode_string(f)
        print(f'{slice_alt_tag=}')
        slice_cell_is_html = get_int(f, 1)
        print(f'{slice_cell_is_html=}')
        slice_cell_text = get_unicode_string(f)
        print(f'{slice_cell_text=}')
        slice_hor_align = get_int(f)
        print(f'{slice_hor_align=}')
        slice_ver_align = get_int(f)
        print(f'{slice_ver_align=}')
        slice_alpha_color = get_int(f, 1)
        slice_red = get_int(f, 1)
        slice_green = get_int(f, 1)
        slice_blue = get_int(f, 1)
        print(f'slice argb={slice_alpha_color}{slice_red}{slice_green}{slice_blue}')

        if f.tell() != end:
            parse_object_descriptor_with_version(f)

def parse_image_resource_slices(f):
    slices_version = get_int(f)
    print(f'{slices_version=}')
    if slices_version in (7,8):
        parse_object_descriptor_with_version(f)
    elif slices_version == 6:
        parse_image_resource_slices_v6(f)

def parse_image_resources_entry(resource_id, f):
    if resource_id == 1050: #slices
        parse_image_resource_slices(f)
    if resource_id == 1005: #slices
        for i in range(2):
            resolution_dpi_int = get_int(f, 2)
            resolution_dpi_frac = get_int(f, 2)
            resolution_dpi = resolution_dpi_int + resolution_dpi_frac / 65536
            dim_type = ('width', 'height')[i]
            print(f'{dim_type} resolution dpi: {resolution_dpi}')
            resolution_dpi_display = get_int(f, 2)
            resolution_dpi_display_str = {1: 'pix/inch', 2: 'pix/cm'}.get(resolution_dpi_display, '???')
            print(f'{dim_type} {resolution_dpi_display=} {resolution_dpi_display_str}')
            resolution_dpi_dim_display = get_int(f, 2)
            resolution_dpi_dim_display_str = {1: 'inches', 2:'cm', 3:'points', 4:'picas', 5:'columns'}.get(resolution_dpi_dim_display, '???')
            print(f'{dim_type} display unit={resolution_dpi_dim_display}, {resolution_dpi_dim_display_str}')

    if resource_id == 1060:
        xml_str = f.read().decode('UTF-8', 'replace')
        xml_str = '\n'.join(l for l in xml_str.splitlines() if l.strip()) # remove huge empty lines blocks
        print(xml_str)

def parse_image_resources_section(f):
    while True:
        sig = f.read(4)
        if not sig:
            break
        assert sig == b'8BIM', repr(sig)
        resource_id = get_int(f, 2)
        print(f'{resource_id=}')

        silent = False
        if (resource_id in skip_global_properties_print):
            silent = True

        name = read_bytestr_padded(f, 2).decode('UTF-8', 'replace')
        if name:
            print(f"name: '{name}'")
        resource_section_entry_size = get_int(f)
        if not silent:
            print(f'{resource_section_entry_size=}')
        data = read_checked(f, resource_section_entry_size)
        read_checked(f, resource_section_entry_size % 2)
        description = image_resource_section_id_description.get(resource_id)
        if not silent:
            print('description:', description)
            print('data:', repr(data))
            if print_readable:
                parse_image_resources_entry(resource_id, io.BytesIO(data))


def psd_parse(f, out_dir):
    header_format = '>4sH6xHIIHH'

    header = read_checked(f, 26)
    assert len(header) == 26, (repr(header[0:50]))
    signature, psd_version, channels, height, width, depth, color_mode = struct.unpack(header_format, header)
    assert signature == b'8BPS', ("Invalid PSD file", repr(header))
    assert psd_version in (1, 2), (psd_version, repr(header)) # 1: PSD, 2: PSB

    print(f"Signature: {signature.decode()}")
    print(f"Version: {psd_version}")
    print(f"Channels: {channels}")
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Depth: {depth}")
    print(f"Color Mode: {color_mode}")

    skip = get_int(f) #Color Mode Data Section
    f.seek(skip, 1)
    print(f'{skip=}')

    image_resources_section_size = get_int(f) #Image Resources Section
    print(f'{image_resources_section_size=}')
    data = read_checked(f, image_resources_section_size)
    parse_image_resources_section(io.BytesIO(data))

    parse_layers(f, psd_version, out_dir)

    parse_global_image(f, psd_version, out_dir, channels, width, height)

    #debug_read_file(f)
    #f.read(80)

def xargs_framework():
    # support for testing on huge set of PSD on multi-CPU processing CPU with xargs.
    # it deals with creation of unique named output folders for each input psd file, and redirecting stdout/stderr to these folders.

    #find ../ -iname  '*.psd' -or -iname '*.psb'  | grep -v '/result.psd'  | xargs -P 6 -d'\n' -n 1 python psd_parse.py

    filename = sys.argv[1]

    import hashlib #pylint: disable=import-outside-toplevel
    import traceback #pylint: disable=import-outside-toplevel
    out_dir = os.path.join('out4', os.path.splitext(os.path.basename(filename))[0] + '-' + hashlib.md5(filename.encode('UTF-8')).hexdigest()[0:4])
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    stderr = sys.stderr
    new_stderr_path = os.path.join(out_dir, 'log_stderr.txt')
    with open(os.path.join(out_dir, 'log_stdout.txt'), 'w', encoding='UTF-8') as new_stdout:
        with open(new_stderr_path, 'w', encoding='UTF-8') as new_stderr:
            try:
                sys.stdout = new_stdout
                sys.stderr = new_stderr

                with open(filename, 'rb') as f:
                    psd_parse(f, out_dir)
            except:
                try:
                    print('error:', filename, file=stderr)
                    print('error:', filename, file=new_stderr)
                    with open(new_stderr_path, 'r', encoding='UTF-8') as new_stderr_finished:
                        stderr.write(new_stderr_finished.read())

                    traceback.print_exc(file = stderr)
                    traceback.print_exc(file = sys.stderr)
                except:
                    pass
                raise

    with open(new_stderr_path, 'r', encoding='UTF-8') as new_stderr_finished:
        stderr.write(new_stderr_finished.read())

    stderr.write('done: ' + filename + '\n')
    stderr.flush()


def main():
    filename = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with open(filename, 'rb') as f:
        psd_parse(f, out_dir)


psd_blend_description = {
    b'pass': 'pass through',
    b'norm': 'normal',
    b'diss': 'dissolve',
    b'dark': 'darken',
    b'mul ': 'multiply',
    b'idiv': 'color burn',
    b'lbrn': 'linear burn',
    b'dkCl': 'darker color',
    b'lite': 'lighten',
    b'scrn': 'screen',
    b'div ': 'color dodge',
    b'lddg': 'linear dodge',
    b'lgCl': 'lighter color',
    b'over': 'overlay',
    b'sLit': 'soft light',
    b'hLit': 'hard light',
    b'vLit': 'vivid light',
    b'lLit': 'linear light',
    b'pLit': 'pin light',
    b'hMix': 'hard mix',
    b'diff': 'difference',
    b'smud': 'exclusion',
    b'fsub': 'subtract',
    b'fdiv': 'divide',
    b'hue ': 'hue',
    b'sat ': 'saturation',
    b'colr': 'color',
    b'lum ': 'luminosity',
}

# copied from documentation https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577411_21585
image_resource_section_id_description = {
 	

1005: 'ResolutionInfo structure. See Appendix A in Photoshop API Guide.pdf',
1006: "Names of the alpha channels as a series of Pascal strings.",
1007: "(Obsolete) See ID 1077DisplayInfo structure. See Appendix A in Photoshop API Guide.pdf.",
1008: "The caption as a Pascal string.",
1009: "Border information. Contains a fixed number (2 bytes real, 2 bytes fraction) for the border width, and 2 bytes for border units (1 = inches, 2 = cm, 3 = points, 4 = picas, 5 = columns).",
1010: 'Background color, ("Color" structure).',
1011: "Print flags. A series of one-byte boolean values (see Page Setup dialog): labels, crop marks, color bars, registration marks, negative, flip, interpolate, caption, print flags.",
1012: "Grayscale and multichannel halftoning information",
1013: "Color halftoning information",
1014: "Duotone halftoning information",
1015: "Grayscale and multichannel transfer function",
1016: "Color transfer functions",
1017: "Duotone transfer functions",
1018: "Duotone image information",
1019: "Two bytes for the effective black and white values for the dot range",
1020: "(Obsolete)",
1021: "EPS options",
1022: "Quick Mask information. 2 bytes containing Quick Mask channel ID; 1- byte boolean indicating whether the mask was initially empty.",
1023: "(Obsolete)",
1024: "Layer state information. 2 bytes containing the index of target layer (0 = bottom layer).",
1025: "Working path (not saved).",
1026: "Layers group information. 2 bytes per layer containing a group ID for the dragging groups. Layers in a group have the same group ID.",
1027: "(Obsolete)",
1028: "IPTC-NAA record. Contains the 'File Info...' information.",
1029: "Image mode for raw format files",
1030: "JPEG quality. Private.",
1032: "(Photoshop 4.0) Grid and guides information.",
1033: "(Photoshop 4.0) Thumbnail resource for Photoshop 4.0 only..",
1034: "(Photoshop 4.0) Copyright flag. Boolean indicating whether image is copyrighted. Can be set via Property suite or by user in File Info...",
1035: "(Photoshop 4.0) URL. Handle of a text string with uniform resource locator. Can be set via Property suite or by user in File Info...",
1036: "(Photoshop 5.0) Thumbnail resource (supersedes resource 1033).",
1037: "(Photoshop 5.0) Global Angle. 4 bytes that contain an integer between 0 and 359, which is the global lighting angle for effects layer. If not present, assumed to be 30.",
1038: "(Obsolete) (Photoshop 5.0) Color samplers resource. Obsolete by id=1073 color samplers format.",
1039: "(Photoshop 5.0) ICC Profile. The raw bytes of an ICC (International Color Consortium) format profile. See ICC1v42_2006-05.pdf ; icProfileHeader.h.",
1040: "(Photoshop 5.0) Watermark. One byte.",
1041: "(Photoshop 5.0) ICC Untagged Profile. 1 byte that disables any assumed profile handling when opening the file. 1 = intentionally untagged.",
1042: "(Photoshop 5.0) Effects visible. 1-byte global flag to show/hide all the effects layer. Only present when they are hidden.",
1043: "(Photoshop 5.0) Spot Halftone. 4 bytes for version, 4 bytes for length, and the variable length data.",
1044: "(Photoshop 5.0) Document-specific IDs seed number. 4 bytes: Base value, starting at which layer IDs will be generated (or a greater value if existing IDs already exceed it). Its purpose is to avoid the case where we add layers, flatten, save, open, and then add more layers that end up with the same IDs as the first set.",
1045: "(Photoshop 5.0) Unicode Alpha Names. Unicode string",
1046: "(Photoshop 6.0) Indexed Color Table Count. 2 bytes for the number of colors in table that are actually defined",
1047: "(Photoshop 6.0) Transparency Index. 2 bytes for the index of transparent color, if any.",
1049: "(Photoshop 6.0) Global Altitude. 4 byte entry for altitude",
1050: "(Photoshop 6.0) Slices.",
1051: "(Photoshop 6.0) Workflow URL. Unicode string",
1052: "(Photoshop 6.0) Jump To XPEP. 2 bytes major version, 2 bytes minor version, 4 bytes count. Following is repeated for count: 4 bytes block size, 4 bytes key, if key = 'jtDd' , then next is a Boolean for the dirty flag; otherwise it's a 4 byte entry for the mod date.",
1053: "(Photoshop 6.0) Alpha Identifiers. 4 bytes of length, followed by 4 bytes each for every alpha identifier.",
1054: "(Photoshop 6.0) URL List. 4 byte count of URLs, followed by 4 byte long, 4 byte ID, and Unicode string for each count.",
1057: "(Photoshop 6.0) Version Info. 4 bytes version, 1 byte hasRealMergedData , Unicode string: writer name, Unicode string: reader name, 4 bytes file version.",
1058: "(Photoshop 7.0) EXIF data 1. See http://www.kodak.com/global/plugins/acrobat/en/service/digCam/exifStandard2.pdf",
1059: "(Photoshop 7.0) EXIF data 3. See http://www.kodak.com/global/plugins/acrobat/en/service/digCam/exifStandard2.pdf",
1060: "(Photoshop 7.0) XMP metadata. File info as XML description. See http://www.adobe.com/devnet/xmp/",
1061: "(Photoshop 7.0) Caption digest. 16 bytes: RSA Data Security, MD5 message-digest algorithm",
1062: "(Photoshop 7.0) Print scale. 2 bytes style (0 = centered, 1 = size to fit, 2 = user defined). 4 bytes x location (floating point). 4 bytes y location (floating point). 4 bytes scale (floating point)",
1064: "(Photoshop CS) Pixel Aspect Ratio. 4 bytes (version = 1 or 2), 8 bytes double, x / y of a pixel. Version 2, attempting to correct values for NTSC and PAL, previously off by a factor of approx. 5%.",
1065: "(Photoshop CS) Layer Comps. 4 bytes (descriptor version = 16), Descriptor structure.",
1066: "(Photoshop CS) Alternate Duotone Colors. 2 bytes (version = 1), 2 bytes count. Count times: [ Color: 2 bytes for space followed by 4 * 2 byte color component ], another 2 byte count, usually 256, by Lab colors one byte each for L,a,b. not read or used by Photoshop.",
1067: "(Photoshop CS)Alternate Spot Colors. 2 bytes (version = 1), 2 bytes channel count, following is repeated for each count: 4 bytes channel ID, Color: 2 bytes for space followed by 4 * 2 byte color component. This resource is not read or used by Photoshop.",
1069: "(Photoshop CS2) Layer Selection ID(s). 2 bytes count, following is repeated for each count: 4 bytes layer ID",
1070: "(Photoshop CS2) HDR Toning information",
1071: "(Photoshop CS2) Print info",
1072: "(Photoshop CS2) Layer Group(s) Enabled ID. 1 byte for each layer in the document, repeated by length of the resource. NOTE: Layer groups have start and end markers",
1073: "(Photoshop CS3) Color samplers resource. Also see ID 1038 for old format.",
1074: "(Photoshop CS3) Measurement Scale. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure)",
1075: "(Photoshop CS3) Timeline Information. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure)",
1076: "(Photoshop CS3) Sheet Disclosure. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure)",
1077: "(Photoshop CS3) DisplayInfo structure to support floating point clors. Also see ID 1007. See Appendix A in Photoshop API Guide.pdf .",
1078: "(Photoshop CS3) Onion Skins. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure)",
1080: "(Photoshop CS4) Count Information. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure) Information about the count in the document. See the Count Tool.",
1082: "(Photoshop CS5) Print Information. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure) Information about the current print settings in the document. The color management options.",
1083: "(Photoshop CS5) Print Style. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure) Information about the current print style in the document. The printing marks, labels, ornaments, etc.",
1084: "(Photoshop CS5) Macintosh NSPrintInfo. Variable OS specific info for Macintosh. NSPrintInfo. It is recommened that you do not interpret or use this data.",
1085: "(Photoshop CS5) Windows DEVMODE. Variable OS specific info for Windows. DEVMODE. It is recommened that you do not interpret or use this data.",
1086: "(Photoshop CS6) Auto Save File Path. Unicode string. It is recommened that you do not interpret or use this data.",
1087: "(Photoshop CS6) Auto Save Format. Unicode string. It is recommened that you do not interpret or use this data.",
1088: "(Photoshop CC) Path Selection State. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure) Information about the current path selection state.",
2000-2997: "Path Information (saved paths).",
2999: "Name of clipping path.",
3000: "(Photoshop CC) Origin Path Info. 4 bytes (descriptor version = 16), Descriptor (Descriptor structure) Information about the origin path data.",
4000-4999: "Plug-In resource(s). Resources added by a plug-in. See the plug-in API found in the SDK documentation",
7000: "Image Ready variables. XML representation of variables definition",
7001: "Image Ready data sets",
7002: "Image Ready default selected state",
7003: "Image Ready 7 rollover expanded state",
7004: "Image Ready rollover expanded state",
7005: "Image Ready save layer settings",
7006: "Image Ready version",
}


main()

