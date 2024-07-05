

'''
Script to convert Clip Studio Paint .clip files to psd.
'''

import re
import io
import sys
import os
import sqlite3
import logging
import struct
import zlib
import traceback
import math
from collections import namedtuple
from functools import cmp_to_key
import itertools
import argparse

#pylint: disable=import-outside-toplevel
# import Image from PIL:
# also imports Image from PIL if command line requires this. module Image is not loaded, if --blank-psd-preview is used, and no output dir for PNGs. So it's possible to export PSD with only built-in python modules.


BlockDataBeginChunk = 'BlockDataBeginChunk'.encode('UTF-16BE')
BlockDataEndChunk = 'BlockDataEndChunk'.encode('UTF-16BE')
BlockStatus = 'BlockStatus'.encode('UTF-16BE')
BlockCheckSum = 'BlockCheckSum'.encode('UTF-16BE')

# returns list of subblocks in rectangular grid of blocks (some are None, for blocks with missing data).
# Subblock data in result list are slices of the input parameter (memoryview)
# rectangular grid is flattened: bitmap_blocks[block_y*grid_width + block_x] has data for (block_x,block_y) cell.
# can return None if can't parse blocks
def parse_chunk_with_blocks(d):
    ii = 0
    block_count1 = 0
    bitmap_blocks = []
    while ii < len(d):
        if d[ii:ii+4+len(BlockStatus)] == b'\0\0\0\x0b' + BlockStatus:
            status_count = int.from_bytes(d[ii+26+4:ii+30+4], 'big')
            if block_count1 != status_count:
                logging.warning("mismatch in block count in layer blocks parsing, %s != %s", block_count1, status_count)
            block_size = status_count * 4 + 12 + (len(BlockStatus)+4)
        elif d[ii:ii+4+len(BlockCheckSum)] == b'\0\0\0\x0d' + BlockCheckSum:
            block_size = 4+len(BlockCheckSum) + 12 + block_count1*4
        elif d[ii+8:ii+8+len(BlockDataBeginChunk)] == BlockDataBeginChunk:
            block_size = int.from_bytes(d[ii:ii+4], 'big')
            expected = b'\0\0\0\x11' + BlockDataEndChunk
            read_data = d[ii+block_size-(4+len(BlockDataEndChunk)):ii+block_size]
            if read_data != expected:
                logging.error("can't parse bitmap chunk, %s != %s", repr(bytes(read_data)), repr(expected))
                return None

            block = d[ii+8+len(BlockDataBeginChunk):ii+block_size-(4+len(BlockDataEndChunk))]
            # 1) first int32 of block contains index of subblock,
            # 2,3,4) then 3 of some unknown in32 parameters,
            # 5) then 0 for empty block or 1 if present.
            # If present, then:
            # 6) size of subblock data in big endian plus 4,
            # 7) then size of subblock in little endian.
            # After these 4*7 bytes actual compressed data follows.

            has_data = int.from_bytes(block[4*4:4*5], 'big')
            if not (0 <= has_data <= 1):
                logging.error("can't parse bitmap chunk (invalid block format, a), %s", repr(has_data))
                return None
            if has_data:
                subblock_len = int.from_bytes(block[5*4:6*4], 'big')
                if not (len(block)  == subblock_len + 4*6):
                    logging.error("can't parse bitmap chunk (invalid block format, b), %s", repr((len(block), subblock_len + 5*6)))
                    return None
                    
                subblock_data = block[7*4:]
                bitmap_blocks.append(subblock_data)
            else:
                bitmap_blocks.append(None)

            block_count1 += 1
        else:
            logging.error("can't recognize %s when parsing subblocks in bitmap layer at %s", repr(d[ii:ii+50]), ii)
            return None

        ii += block_size
    if len(d) != ii:
        logging.warning("invalid last block size, overflow %s by %s", len(d), ii)
    return bitmap_blocks

ChunkInfo = namedtuple("ChunkInfo", ("layer_str", "chunk_info_filename", "bitmap_blocks"))

def iterate_file_chunks(data, filename):
    file_header_size = 24
    file_header = data[0:file_header_size]
    if not(b'CSFCHUNK' == file_header[0:8]):
        raise ValueError(f"can't recognize Clip Studio file '{filename}', {repr(data[0:30])}")
    chunk_offset = file_header_size
    t = data[file_header_size:file_header_size+4] 
    if not (t == b'CHNK'):
        raise ValueError(f"can't find first chunk in Clip Studio file after header, '{filename}', {repr(t)}")

    file_chunks_list = []

    while chunk_offset < len(data):
        t = data[chunk_offset:chunk_offset+4] 
        if not (t == b'CHNK'):
            raise ValueError(f"can't find next chunk in Clip Studio file after header, '{filename}', {repr(t)}")
        chunk_header = data[chunk_offset:chunk_offset+4*4]
        chunk_name = chunk_header[4:8]
        zero1 = chunk_header[8:12]
        size_bin = chunk_header[12:16]
        if zero1 != b'\0'*4: 
            logging.warning('interesting, not zero %s %s %s', repr(chunk_name), filename, repr(zero1))
        chunk_data_size = int.from_bytes(size_bin, 'big')

        chunk_data_memory_view = memoryview(data)[chunk_offset+16:chunk_offset+16+chunk_data_size]
        file_chunks_list.append( (chunk_name, chunk_data_memory_view, chunk_offset) )

        chunk_offset += 16 + chunk_data_size

    return file_chunks_list

def extract_csp_chunks_data(file_chunks_list, out_dir, chunk_to_layers, layer_names):
    if out_dir:
        for f in os.listdir(out_dir):
            if f.startswith('chunk_'):
                os.unlink(os.path.join(out_dir, f))

    chunks = {}

    for chunk_name, chunk_data_memory_view, chunk_offset in file_chunks_list:
        chunk_data_size = len(chunk_data_memory_view)
        if chunk_name == b'Exta':
            chunk_name_length = int.from_bytes(chunk_data_memory_view[:+8], 'big')
            if not chunk_name_length == 40:
                logging.warning("warning, unusual chunk name length=%s, usualy it's 40", chunk_name_length)
            chunk_id = bytes(chunk_data_memory_view[8:8+chunk_name_length])
            if chunk_id[0:8] != b'extrnlid':
                logging.warning('%s', f"warning, unusual chunk name, expected name starting with 'extrnlid' {repr(chunk_id)}")

            logging.debug('%s '*7, chunk_name.decode('ascii'), 'chunk_data_size:', chunk_data_size, 'offset:', chunk_offset, 'id:', chunk_id.decode('ascii'))
            chunk_size2 = int.from_bytes(chunk_data_memory_view[chunk_name_length+8:chunk_name_length+8+8], 'big')
            if not(chunk_data_size == chunk_size2 + 16 + chunk_name_length ):
                logging.warning('%s', f"warning, unusual second chunk size value, expected ({chunk_data_size=}) = ({chunk_size2=}) + 16 + ({chunk_name_length=}) ")
            
            chunk_binary_data = chunk_data_memory_view[chunk_name_length+8+8:]

            bitmap_blocks = None
            if chunk_binary_data[8:8+len(BlockDataBeginChunk)] == BlockDataBeginChunk:
                bitmap_blocks = parse_chunk_with_blocks(chunk_binary_data)
                if bitmap_blocks == None:
                    logging.error("can't parse bitmap id=%s block at %s", repr(chunk_id), chunk_offset)
                    continue
                    
                ext='png'
            else:
                ext = 'bin'

            # vector data export is not implemented, so disable it's extraction
            #elif 'vector' in chunk_main_types:
            #    chunk_txt_info = io.StringIO()
            #    k = 22*4
            #    for i in range(1 + (len(chunk_binary_data)-1) // k):
            #        print(chunk_binary_data[i*k:(i+1)*k].hex(' ', 4), file=chunk_txt_info)
            #    chunk_output_data = chunk_txt_info.getvalue().encode('UTF-8')

            if not re.match(b'^[a-zA-Z0-9_-]+$', chunk_id):
                logging.warning("unusual chunk_id=%s", repr(chunk_id))

            layer_num_str = '_'.join(f'{id:03d}' for id in chunk_to_layers.get(chunk_id, []))
            def make_layer_name_for_file(id):
                layer_name = layer_names.get(id, f'no-name-{id}').strip()
                if not layer_name:
                    layer_name = '{empty}'
                layer_name_sanitized = re.sub(r'''[\0-\x1f'"/\\:*?<>| ]''', '_', layer_name)[0:80]
                return layer_name_sanitized

            layer_name_str = ','.join(make_layer_name_for_file(id) for id in chunk_to_layers.get(chunk_id, []))
            layer_name_str = '[' + layer_name_str + ']'

            chunk_info_filename = f'chunk_{layer_num_str}_{chunk_id.decode("ascii")}_{layer_name_str}.{ext}'
            # side effect - save non-bitmape binary data chunks information
            if chunk_binary_data != None and out_dir and not bitmap_blocks:
                with open(os.path.join(out_dir, chunk_info_filename), 'wb') as f:
                    f.write(chunk_binary_data)

            chunks[chunk_id] = ChunkInfo(layer_name_str, chunk_info_filename, bitmap_blocks)
        else:
            logging.debug('%s '*5, chunk_name.decode('ascii'), 'chunk_data_size:', chunk_data_size, 'offset:', chunk_offset)

    return chunks


def sort_tuples_with_nones(tuples):
    def cmp_tuples_with_none(aa, bb):
        assert len(aa) == len(bb)
        for a, b in zip(aa, bb):
            if a != b:
                if a == None:
                    return -1
                if b == None:
                    return 1
                return (-1 if (a < b) else +1)
        return 0 # equal tuples
    return sorted(tuples, key = cmp_to_key(cmp_tuples_with_none))


# Make sql query and return results as named tuples
def execute_query_global(conn, query, namedtuple_name = "X"):
    cursor = conn.cursor()
    cursor.execute(query)
    # get column names (nameduple forbids underscore '_' at start of name).
    column_names = [description[0].removeprefix('_') for description in cursor.description]
    table_row_tuple_type = namedtuple(namedtuple_name, column_names)
    # Fetch all results and convert them to named tuples

    results = [table_row_tuple_type(*row) for row in cursor.fetchall()]
    return sort_tuples_with_nones(results)# easier to keep stability in stdout and reduce diff clutter


def one_column(rows):
    assert all(len(row) == 1 for row in rows)
    return [row[0] for row in rows]


def dump_database_chunk_links_structure_info(conn):
    refs_by_main_id = execute_query_global(conn, "SELECT TableName, LabelName, LinkTable from ParamScheme WHERE LinkTable <> ''", 'TableLinkScheme')
    refs_by_external_id = execute_query_global(conn, "SELECT TableName, ColumnName from ExternalTableAndColumnName", "ExternalTableAndColumnName")

    with open('x.dot', 'w', encoding='UTF-8') as f: # "cat x.dot | dot -Tpng -o x.png" to get picture
        print('digraph {', file=f)
        #reachable = {'Offscreen'}
        reachable = {ref.TableName for ref in refs_by_external_id}

        found_new = True
        while found_new:
            found_new = False
            for p in refs_by_main_id:
                if p.LinkTable in reachable and p.TableName not in reachable:
                    reachable.add(p.TableName)
                    found_new = True

        for ref in refs_by_external_id:
            print(f'{ref.TableName} [ label="{ref.TableName} | {ref.ColumnName}", shape=rectangle, style=filled, fillcolor=lightgray, fontname="Helvetica"]', file=f)
        group_arrows = {}
        for p in refs_by_main_id:
            if p.TableName in reachable and p.LinkTable in reachable:
                group_arrows.setdefault((p.TableName, p.LinkTable), []).append(p.LabelName)
        for (TableName, LinkTable), arrow_labels in sorted(group_arrows.items()):
            if TableName != "Project":
                arrow_label = '\n'.join(arrow_labels)
                print(f'{TableName} -> {LinkTable} [label="{arrow_label}", fontname="Helvetica"]', file=f)
        for p in refs_by_main_id:
            if p.TableName in reachable and p.LinkTable in reachable:
                if p.TableName == "Project":
                    print(f'{p.TableName} -> "{p.LinkTable}..." [label="{p.LabelName}", fontname="Helvetica"]', file=f)
        print('}', file=f)


def get_database_columns(conn):
    tables = one_column(execute_query_global(conn, "SELECT name FROM sqlite_schema WHERE type == 'table' ORDER BY name"))

    table_columns = {}
    for t in tables:
        if not re.match('^[a-zA-Z0-9_.]+$', t):
            continue
        table_columns[t] = one_column(execute_query_global(conn, f"SELECT name FROM pragma_table_info('{t}')  WHERE type <> '' order by name"))

    return table_columns

def get_sql_data_layer_chunks():
    db_path = os.path.join(cmd_args.sqlite_file)

    query_offscreen_chunks = 'SELECT MainId, LayerId, BlockData, Attribute from Offscreen;'  # LayerId is used to have layer id for layer chunk types I have no interest (thumbs, smaller mipmaps)
    # it's easier to SELECT */getattr than to try to deal with non-existing columns with exactly same result.
    # "FilterLayerInfo" is optional, maybe some other fields are optional too.
    #layer_attributes = 'MainId, CanvasId, LayerName, LayerType, LayerComposite, LayerOpacity, LayerClip, FilterLayerInfo,
    #    LayerLayerMaskMipmap, LayerRenderMipmap,
    #    LayerVisibility, LayerLock,  LayerMasking,
    #    LayerOffsetX, LayerOffsetY, LayerRenderOffscrOffsetX, LayerRenderOffscrOffsetY,
    #    LayerMaskOffsetX, LayerMaskOffsetY, LayerMaskOffscrOffsetX, LayerMaskOffscrOffsetY,
    #    LayerSelect, LayerFirstChildIndex, LayerNextIndex'.replace(',', ' ').split()
    #
    #    table_columns_lowercase = { key.lower() : [c.lower() for c in val] for key, val in table_columns }
    #    layer_existing_columns = table_columns_lowercase['layer']
    #    layer_sql_query_columns = ','.join( x if x.lower() in layer_extra_section else 'NULL' for layer_attributes )
    #    query_layer = f'SELECT {layer_sqlite_info} FROM Layer'
    query_layer = 'SELECT * FROM Layer;'

    query_mipmap = 'SELECT MainId, BaseMipmapInfo from Mipmap'
    query_mipmap_info = 'SELECT MainId, Offscreen from MipmapInfo' # there is NextIndex to get mipmap chain information, but lower mipmpas are not needed for export.
    query_vector_chunks = 'SELECT MainId, VectorData, LayerId from VectorObjectList'

    with sqlite3.connect(db_path) as conn:
        table_columns = get_database_columns(conn)

        def execute_query(conn, query, namedtuple_name,  optional_table = None):
            if optional_table:
                if optional_table not in table_columns:
                    return []
            return execute_query_global(conn,  query, namedtuple_name)

        offscreen_chunks_sqlite_info = execute_query(conn, query_offscreen_chunks, 'OffscreenChunksTuple')
        layer_sqlite_info = execute_query(conn, query_layer, 'LayerTuple')
        mipmap_sqlite_info = execute_query(conn, query_mipmap, 'MipmapChainHeader')
        mipmapinfo_sqlite_info = execute_query(conn, query_mipmap_info, 'MipmapLevelInfo')
        vector_info = execute_query(conn, query_vector_chunks, 'VectorChunkTuple', optional_table = "VectorObjectList")
        #dump_database_chunk_links_structure_info(conn)

        #pylint: disable=too-many-instance-attributes
        class SqliteInfo:
            def __init__(self):
                self.offscreen_chunks_sqlite_info = offscreen_chunks_sqlite_info
                self.layer_sqlite_info = layer_sqlite_info
                self.mipmap_sqlite_info = mipmap_sqlite_info
                self.mipmapinfo_sqlite_info = mipmapinfo_sqlite_info
                self.vector_info = vector_info
                self.canvas_preview_data = one_column(execute_query_global(conn, 'SELECT ImageData FROM CanvasPreview'))
                self.root_folder = one_column(execute_query_global(conn, 'SELECT CanvasRootFolder FROM Canvas'))[0]
                self.width, self.height, self.dpi = execute_query_global(conn, 'SELECT CanvasWidth, CanvasHeight, CanvasResolution from Canvas')[0]
        result = SqliteInfo()
    conn.close()
    return result

def read_csp_int(f):
    t = f.read(4)
    assert len(t) == 4, (len(t), "unexpected end of data section")
    return int.from_bytes(t, 'big')

def read_csp_double(f):
    return struct.unpack('>d', f.read(8))[0]

def read_csp_int_maybe(f):
    t = f.read(4)
    if len(t) != 4:
        return None
    return int.from_bytes(t, 'big')

def read_csp_unicode_str(f):
    str_size = read_csp_int_maybe(f)
    if str_size == None:
        return None
    string_data = f.read(2 * str_size)
    return string_data.decode('UTF-16-BE')

def parse_offscreen_attributes_sql_value(offscreen_attribute):
    b_io = io.BytesIO(offscreen_attribute)
    def get_next_int(): return int.from_bytes(b_io.read(4), 'big')
    def check_read_str(s):
        s2 = read_csp_unicode_str(b_io)
        assert s2 == s

    header_size = get_next_int()
    assert header_size == 16, (header_size, repr(offscreen_attribute))
    info_section_size = get_next_int()
    assert info_section_size == 102
    extra_info_section_size = get_next_int()
    assert extra_info_section_size in (42, 58), (extra_info_section_size, repr(offscreen_attribute))
    get_next_int()

    check_read_str("Parameter")
    bitmap_width = get_next_int()
    bitmap_height = get_next_int()
    block_grid_width = get_next_int()
    block_grid_height = get_next_int()

    attributes_arrays = [get_next_int() for _i in range(16)]

    check_read_str("InitColor")
    get_next_int()
    default_fill_black_white = get_next_int()
    get_next_int()
    get_next_int()

    get_next_int()

    init_color = [0] * 4
    if (extra_info_section_size == 58):
        init_color = [min(255, get_next_int() // (256**3)) for _i in range(4)]

    return [
        bitmap_width, bitmap_height,
        block_grid_width, block_grid_height,
        default_fill_black_white,
        attributes_arrays,
        init_color
    ]


def rle_compress(line, n, max_seg, buf_dst):
    # works 3x faster, but produce 20x sized multi-gigabyte psd
    fast = False
    if fast:
        dst = 0
        i = 0
        while i < n:
            l = min(128, n - i)
            buf_dst[dst] = l - 1
            dst += 1
            buf_dst[dst:dst+l] = line[i:i+l]
            dst += l
            i += l
        return dst

    #heuristic for RLE:
    # It's always worth to compress 3 equal bytes as RLE, it never makes worse. Split input by sequences of 3 or more equal consequtive bytes ("long run"), handle ranges between.
    dst = 0
    assert max_seg >= 3, (max_seg) # value less than 3 brokes logic of long run heuristic and makes RLE meaningless

    def between_long_run(start, end, dst):
        # compress couples of equal bytes at start as RLE, fit remainder to raw section.
        # It's not perfectly optimal, but almost optimal: it's always optimal to extend raw section with 2 more equal bytes instead starting new RLE.
        # Except case when it grow to 128 bytes RLE limit. In this case 1 bytes economy doesn't matter much, even rare cases when possible.
        # There is always no more than couple of bytes available for RLE, because 3 bytes and longer are handled in caller loop.
        i = start
        while end - i >= 2:
            c = line[i]
            if c != line[i+1]:
                break
            i += 2
            buf_dst[dst] = 255
            buf_dst[dst+1] = c
            dst += 2

        end2 = end
        while end - i >= 2:
            c = line[end-1]
            c2 = line[end-2]
            if c != c2:
                break
            end -= 2

        while i < end:
            d = min(max_seg, end - i)
            buf_dst[dst] = d - 1
            dst += 1
            buf_dst[dst:dst+d] = line[i:i+d]
            dst += d
            i += d

        while end < end2:
            c = line[end]
            buf_dst[dst] = 255
            buf_dst[dst+1] = c
            end += 2
            dst += 2

        return dst

    def write_long_run(length, c, dst):
        while length > 0:
            d = min(length, max_seg)
            buf_dst[dst] = 257 - d
            buf_dst[dst+1] = c
            dst += 2
            length -= d
        return dst

    def write_raw_and_rle(equal_run, prev_long_run_end, i, equal_val, dst):
        #print(f'{equal_run=}, {prev_long_run_end=}, {i=}')
        if equal_run > max_seg and equal_run % max_seg == 1:
            equal_run -= 1 # remainder of length 1 cannot be written as rle because rle is always 2 or larger, so merge remainder of size 1 to previous raw section
        dst = between_long_run(prev_long_run_end, i - equal_run, dst)
        dst = write_long_run(equal_run, equal_val, dst)
        return dst

    if n == 0:
        return 0

    special_handling_of_first_equal_run = True
    if special_handling_of_first_equal_run:
        i = 1
        first = line[0]
        while i < n and line[i] == first:
            i += 1
        if i > max_seg and i % max_seg == 1:
            # remainder of length 1 cannot be written as rle because rle is always 2 or larger, and there is no previos raw section to merge the remainder. So, leave it to next sections.
            i -= 1
        prev_long_run_end = 0
        if i >= 3:
            dst = write_long_run(i, first, dst)
            prev_long_run_end = i
            if i == n:
                return dst
        if i == n:
            return between_long_run(0, n, dst)
        equal_run = 1
        prev = line[i]
    else:
        equal_run = 0
        prev_long_run_end = 0
        prev = None
        i = -1

    for i in range(i+1, n):
        cur = line[i]
        if cur != prev:
            if equal_run >= 3:
                dst = write_raw_and_rle(equal_run, prev_long_run_end, i, prev, dst)
                prev_long_run_end = i
            equal_run = 1
        else:
            equal_run += 1
        prev = cur

    if equal_run >= 3:
        dst = write_raw_and_rle(equal_run, prev_long_run_end, n, prev, dst)
    else:
        dst = between_long_run(prev_long_run_end, n, dst)

    return dst


def join_rle_scanlines_to_psd_channel(lines, channel_output_tmp_buf, psd_version):
    i = 0
    compression_type = 1
    channel_output_tmp_buf[0:2] = int.to_bytes(compression_type, 2, 'big')
    i += 2
    size_of_size = psd_version*2
    sizes_array_start = i
    sizes_array_size = len(lines) * size_of_size
    i += sizes_array_size
    for i_line, result_line_pieces in enumerate(lines):
        output_line_size = 0
        for piece in result_line_pieces:
            piece_size = len(piece)
            output_line_size += piece_size
            channel_output_tmp_buf[i:i+piece_size] = piece
            i += piece_size
        k = sizes_array_start + i_line*size_of_size
        channel_output_tmp_buf[k:k+size_of_size] = int.to_bytes(output_line_size, size_of_size, 'big')
    return channel_output_tmp_buf[0:i]

def decode_to_psd_rle(offscreen_attribute, bitmap_blocks, psd_version, img_type):
    parsed_offscreen_attributes = parse_offscreen_attributes_sql_value(offscreen_attribute)
    bitmap_width, bitmap_height, block_grid_width, block_grid_height, default_fill_black_white, pixel_packing_params, _init_color = parsed_offscreen_attributes

    first_packing_channel_count = pixel_packing_params[1]
    second_packing_channel_count = pixel_packing_params[2]
    packing_type = (first_packing_channel_count, second_packing_channel_count)
    channel_count_sum = sum(packing_type)
    default_one_channel_color = default_fill_black_white * 255

    bit_packing = pixel_packing_params[8] == 32
    assert not bit_packing, "unpack of 1-bit is not implemented yet" # caused by external file format, still more like problem of the script

    assert block_grid_width * block_grid_height == len(bitmap_blocks)

    existing_blocks = [(j, i) for i in range(block_grid_height) for j in range(block_grid_width) if bitmap_blocks[i*block_grid_width + j]]

    if existing_blocks:
        j_set = [j for (j, i) in existing_blocks]
        i_set = [i for (j, i) in existing_blocks]

        j_start = min(j_set)
        j_end = max(j_set) + 1

        i_start = min(i_set)
        i_end = max(i_set) + 1

        bitmap_grid_width =  (j_end - j_start)*256
        bitmap_grid_height = (i_end - i_start)*256
        bitmap_offset_x = j_start*256
        bitmap_offset_y = i_start*256
    else:
        bitmap_offset_x = 0
        bitmap_offset_y = 0
        j_start = j_end = 0
        i_start = i_end = 0

    output_bitmap_width = min(j_end * 256, bitmap_width) - bitmap_offset_x
    output_bitmap_height = min(i_end * 256, bitmap_height) - bitmap_offset_y

    k = 256*256

    if img_type == "layer":
        assert packing_type == (1, 4)
        channel_definition = [(0, 1, -1), (k+2, 4, 0), (k+1, 4, 1), (k+0, 4, 2)] # a, folowed by interleaved (b, g, r, unused)
    elif img_type == "mask":
        assert channel_count_sum == 1
        mask_psd_tag = -2
        channel_definition = [(0, 1, mask_psd_tag)]
    else:
        assert False, repr(img_type)

    # offset, multiply, psd_channel_tag
    channel_scanlines = [
        [ [] for _i_line in range(output_bitmap_height) ]
        for _ in channel_definition
    ]

    # "packbits" RLE format empty pieces:
    def empty_n_limited_128(n):
        assert n > 0
        if n == 1:
            return bytearray((0, default_one_channel_color)) #raw 1-byte section, because RLE can't contain just 1 byte
        return bytearray((257 - n, default_one_channel_color))
    empty128 = empty_n_limited_128(128)
    empty256 = empty128 + empty128
    def empty_n_limited_256(n):
        assert 0 < n <= 256
        if n == 256:
            return empty256 # optimization shortcut - empty 256x256 blocks contains bunch of 256 empty pieces
        if n <= 128:
            return empty_n_limited_128(n)
        return empty128 + empty_n_limited_128(n % 128)

    rle_output_tmp_buf = bytearray(2*output_bitmap_width) # temporary buffer with safety margin for worst possible compression
    rle_input_tmp_buf = bytearray(output_bitmap_width) 

    for i in range(i_start, i_end):
        block_grid_line = [None] * (j_end - j_start)
        for j in range(j_start, j_end):
            dj = j - j_start
            di = i - i_start

            block = bitmap_blocks[i*block_grid_width + j]
            if block:
                try:
                    pixel_data_bytes = zlib.decompress(block)
                except:
                    pixel_data_bytes = None
                    if not cmd_args.ignore_zlib_errors:
                        logging.error("can't decompress pixel data block")
                        raise
                    logging.info("can't decompress pixel data block, skipped")

                if pixel_data_bytes != None:
                    pixel_data = memoryview(pixel_data_bytes)
                    block_grid_line[dj] = pixel_data

        def is_non_empty_block(p):
            empty_block =  p[1] == None or cmd_args.psd_empty_bitmap_data
            return not empty_block

        for exists, block_data_group in itertools.groupby(enumerate(block_grid_line), is_non_empty_block):
            block_data_group = list(block_data_group)
            for i_channel, (channel_offset, multiply, _psd_channel_tag) in enumerate(channel_definition):
                for i_line in range(256):
                    i_image_line = i_line + di*256
                    if i_image_line >= output_bitmap_height:
                        break
                    channel_scanline_output = channel_scanlines[i_channel][i_image_line]

                    if not exists:
                        for dj, _empty_block_data in block_data_group:
                            #todo: create test files with empty blocks of 1 and 256 sizes in the row/column of 3x3 and 4x4 grid (768x768 and and 769x769 files)

                            if (dj+1)*256 <= output_bitmap_width:
                                channel_scanline_output.append(empty256)
                            else:
                                channel_scanline_output.append(empty_n_limited_256(output_bitmap_width - dj*256))
                    else:
                        i_write = 0
                        for dj, block_data in block_data_group:
                            channel_memory_view = block_data[channel_offset:][::multiply][i_line*256:]
                            if (dj+1)*256 > output_bitmap_width:
                                block_width = output_bitmap_width - dj*256
                            else:
                                block_width = 256
                            rle_input_tmp_buf[i_write:i_write + block_width] = channel_memory_view[:block_width]
                            i_write += block_width
                        compressed_size = rle_compress(rle_input_tmp_buf, i_write, 128, rle_output_tmp_buf)
                        channel_scanline_output.append(rle_output_tmp_buf[0:compressed_size])

    #todo: crop margins (take in account that raw section can contain 1/2 similar bytes) #TODO: better do this before, it's hard to strip mixed rle/raw sections of multichannel. shortcut - zero min stop iteration.
    # 2 for compression type, 2 for worst possible rle compression, 4 for scanline size
    channel_output_tmp_buf = bytearray(2 + (output_bitmap_width*2 + 4)*output_bitmap_height)

    assert len(channel_definition) == len(channel_scanlines)
    channel_scanlines = [
        (psd_channel_tag, join_rle_scanlines_to_psd_channel(output_scanline_data, channel_output_tmp_buf, psd_version))
        for (_, _, psd_channel_tag), output_scanline_data in zip(channel_definition, channel_scanlines)
    ]

    return channel_scanlines, bitmap_offset_x, bitmap_offset_y, output_bitmap_width, output_bitmap_height, default_one_channel_color

def decode_to_img(offscreen_attribute, bitmap_blocks): 
    from PIL import Image

    parsed_offscreen_attributes = parse_offscreen_attributes_sql_value(offscreen_attribute)
    bitmap_width, bitmap_height, block_grid_width, block_grid_height, default_fill_black_white, pixel_packing_params, _init_color = parsed_offscreen_attributes

    first_packing_channel_count = pixel_packing_params[1]
    second_packing_channel_count = pixel_packing_params[2]
    packing_type = (first_packing_channel_count, second_packing_channel_count)
    channel_count_sum = sum(packing_type)
    assert packing_type == (1, 4) or (channel_count_sum == 1), packing_type
    assert block_grid_width * block_grid_height == len(bitmap_blocks)

    if packing_type == (1, 4):
        default_fill = (255,255,255,255) if default_fill_black_white else (0,0,0,0)
        img = Image.new("RGBA", (bitmap_width, bitmap_height), default_fill)
    else:
        assert channel_count_sum == 1
        default_fill = 255 if default_fill_black_white else 0
        img = Image.new("L", (bitmap_width, bitmap_height), default_fill)

    for i in range(block_grid_height):
        for j in range(block_grid_width):
            block = bitmap_blocks[i*block_grid_width + j]
            if block:
                try:
                    pixel_data_bytes = zlib.decompress(block)
                except:
                    if not cmd_args.ignore_zlib_errors:
                        logging.error("can't unpack block data with zlib, --ignore-zlib-errors can be used to ignore errors")
                        raise
                    else:
                        logging.debug("can't unpack block data with zlib")
                    continue
                pixel_data = memoryview(pixel_data_bytes)
                k = 256*256
                if packing_type == (1, 4):
                    if len(pixel_data) != 5*k:
                        logging.error("invalid pixel count for 4-channel block, expected 5*256*256, got %s", len(pixel_data))
                        continue
                    block_img_alpha = Image.frombuffer("L", (256, 256), pixel_data[0:k], 'raw')
                    block_img_rgbx = Image.frombuffer("RGBA", (256, 256), pixel_data[k:5*k], 'raw')
                    b,g,r, _ = block_img_rgbx.split()
                    a, = block_img_alpha.split()
                    block_result_img = Image.merge("RGBA", (r,g,b,a))
                else:
                    if len(pixel_data) != k:
                        logging.error("invalid pixel count for 1-channel block, expected 256*256, got %s", len(pixel_data))
                        continue
                    # this branch won't run until masks be saved as png
                    block_result_img = Image.frombuffer("L", (256, 256), pixel_data[0:k], 'raw')
                img.paste(block_result_img, (256*j, 256*i))
    return img

def decode_layer_to_png(offscreen_attribute, bitmap_blocks):
    img = decode_to_img(offscreen_attribute, bitmap_blocks)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='png', compress_level=1)
    return img_byte_arr.getvalue()


def save_layers_as_png(chunks, out_dir, sqlite_info):
    #ComicFrameLineMipmap LayerLayerMaskMipmap LayerRenderMipmap ResizableOriginalMipmap TimeLineOriginalMaskMipmap TimeLineOriginalMipmap"
    mipmapinfo_dict = { m.MainId:m for m in sqlite_info.mipmapinfo_sqlite_info }
    mipmap_dict     = { m.MainId:m for m in sqlite_info.mipmap_sqlite_info }
    offscreen_dict  = { m.MainId:m for m in sqlite_info.offscreen_chunks_sqlite_info }

    referenced_chunks_data = {}

    for l in sqlite_info.layer_sqlite_info:
        mipmap_id = l.LayerRenderMipmap
        if mipmap_id != None:
            external_block_row = offscreen_dict[mipmapinfo_dict[mipmap_dict[mipmap_id].BaseMipmapInfo].Offscreen]
            external_id = external_block_row.BlockData
            chunk_info = chunks.get(external_id)
            if chunk_info != None:
                #c.layer_name
                referenced_chunks_data[external_id] = (external_block_row.Attribute, chunk_info)
                #offscreen_chunks_sqlite_info.setdefault(external_id, []).append(l.MainId)

    for external_id, (offscreen_attribute, chunk_info) in sorted(referenced_chunks_data.items()):
        png_data = decode_layer_to_png(offscreen_attribute, chunk_info.bitmap_blocks)
        chunk_info_filename = chunks[external_id].chunk_info_filename
        assert chunk_info_filename.endswith('.png')
        logging.info(os.path.join(out_dir, chunk_info_filename))
        with open(os.path.join(out_dir, chunk_info_filename), 'wb') as f:
            f.write(png_data)


LayerBitmaps = namedtuple("LayerBitmaps", ["LayerBitmap", "LayerMaskBitmap"])
def get_layers_bitmaps(chunks, sqlite_info):
    #ComicFrameLineMipmap LayerLayerMaskMipmap LayerRenderMipmap ResizableOriginalMipmap TimeLineOriginalMaskMipmap TimeLineOriginalMipmap"
    mipmapinfo_dict = { m.MainId:m for m in sqlite_info.mipmapinfo_sqlite_info }
    mipmap_dict     = { m.MainId:m for m in sqlite_info.mipmap_sqlite_info }
    offscreen_dict  = { m.MainId:m for m in sqlite_info.offscreen_chunks_sqlite_info }

    layer_bitmaps = { }

    def get_layer_chunk_data(l, mipmap_id):
        if mipmap_id:
            external_block_row = offscreen_dict[mipmapinfo_dict[mipmap_dict[mipmap_id].BaseMipmapInfo].Offscreen]
            external_id = external_block_row.BlockData
            if external_id not in chunks:
                logging.debug("layer %s references non-existing chunk %s", repr(l.LayerName), repr(external_id))
                return None
            return (chunks[external_id].bitmap_blocks, external_block_row.Attribute)
        return None

    for l in sqlite_info.layer_sqlite_info:
        layer_bitmaps[l.MainId] = LayerBitmaps(
            get_layer_chunk_data(l, l.LayerRenderMipmap),
            get_layer_chunk_data(l, l.LayerLayerMaskMipmap))

    return layer_bitmaps


def escape_bytes_str(s):
    # escaping binary strings with non-ascii characters. Similar to urllib.parse.quote, but
    # with more stable and trivial 'safe' characters definition

    percent = ord('%')

    if all(32 <= x < 128 and x != percent for x in s):
        return s

    def hexdigit_char(i):
        assert 0 <= i < 16
        if i < 10:
            return 48 + i  # ord('0')
        return 55 + i # ord('A') - 10

    def byte2hex(s):
        for x in s:
            if 32 <= x < 128 and x != percent:
                yield x
            else:
                yield percent
                yield hexdigit_char(x // 16)
                yield hexdigit_char(x % 16)
    return bytes(byte2hex(s))



class DataReader:
    def __init__(self, data):
        self.data = data
        self.ofs = 0

    def size(self):
        return len(self.data)

    def left(self):
        return len(self.data) - self.ofs

    def unpack_int(self, size, byteorder, signed = False):
        i = self.ofs
        self.ofs += size
        return int.from_bytes(self.data[i:self.ofs], byteorder, signed=signed)

    def read_n(self, length):
        result = self.data[self.ofs:self.ofs+length]
        self.ofs += length
        return result

    def read_int32_le(self, signed = False):
        return self.unpack_int(4, 'little', signed)

    def read_int16_le(self):
        return self.unpack_int(2, 'little')

    def read_int8_le(self):
        return self.unpack_int(1, 'little')

    def read_int32_be(self, signed = False):
        return self.unpack_int(4, 'big', signed)

    def read_int16_be(self):
        return self.unpack_int(2, 'big')

    def read_int8_be(self):
        return self.unpack_int(1, 'big')

    def read_float64(self):
        value = struct.unpack_from('<d', self.data, self.ofs)[0]
        self.ofs += 8
        return value

    def read_string(self, length, encoding='utf-8'):
        value = self.data[self.ofs:self.ofs + length].decode(encoding)
        self.ofs += length
        return value

#pylint: disable=too-many-return-statements
def parse_layer_text_attribute_param(param_id, data_reader):
    #pylint: disable=attribute-defined-outside-init
    #pylint: disable=too-many-instance-attributes
    class Obj:
        def __repr__(self):
            d = self.__dict__
            prefix = ''
            if 'length' in self.__dict__ and 'start' in self.__dict__:
                prefix = f'[{self.start} ; + {self.length} ; --> {self.start+self.length}]'
                d = dict(d)
                del d['length']
                del d['start']
            return prefix + ' ' + str(''.join(f'{a}: {repr(b)}; ' for a, b in sorted(d.items())))

    param_data_size = data_reader.size()
    param_value = []


    if param_id == 11:
        n = data_reader.read_int32_le()
        param_value = []
        for _i in range(n):
            run = Obj()
            run.start = data_reader.read_int32_le(signed = True)
            run.length = data_reader.read_int32_le()
            entry_size = data_reader.read_int32_le()
            end_of_runs_data = data_reader.ofs + entry_size - 8
            run.style_flags = data_reader.read_int8_le()
            run.field_defaults_flags = data_reader.read_int8_le()
            run.color = [data_reader.read_int16_le() / 65535 for _ in range(3)]
            run.font_scale = data_reader.read_float64()
            str_len = data_reader.read_int16_le()
            run.font = data_reader.read_string(str_len*2, 'UTF-16LE')
            param_value.append(run)
            if data_reader.ofs != end_of_runs_data:
                logging.warning("unexpected value of entry_size, %s", entry_size)
            data_reader.ofs = max(data_reader.ofs, end_of_runs_data)
        return 'runs', param_value, None

    if param_id in (12, 16, 20):
        param_value = []
        size = data_reader.read_int32_le()
        for _i in range(size):
            item = Obj()
            item.start = data_reader.read_int32_le(signed = True)
            item.length = data_reader.read_int32_le()
            unk1_32 = data_reader.read_int32_le()
            item.align = data_reader.read_int8_le()
            unk2_8 = data_reader.read_int8_le()
            param_value.append(item)
        param_name = {12: 'param_align', 16: 'param_underline', 20:'param_strike'}[param_id]
        return  param_name, param_value, [ unk2_8, unk1_32 ]
    if param_id == 31:
        return 'font', data_reader.read_string(param_data_size, "UTF-8"), None
    if param_id == 32:
        return 'font_size', data_reader.read_int32_le(), None
    if param_id == 26:
        p = data_reader.read_n(param_data_size)
        if not(len(p) >= 32):
            logging.error("can't read aspect_ratio for text layer settings (too small data), assume 1.0 as default")
            return 'aspect_ratio', 1.0, None
        dd = struct.unpack('<dd', p[16:32])
        return 'aspect_ratio', dd[0], dd
    if param_id == 34:
        param_value = []
        color_ints = [data_reader.read_int32_le() for _i in range(3)]
        param_value = [c/(2**32-1) for c in color_ints]
        return 'color', param_value, color_ints
    if param_id == 42:
        param_value = [data_reader.read_int32_le(signed=True) for _i in range(4)]
        return 'bbox', param_value, None
    if param_id == 32: 
        return 'font_size', data_reader.read_int32_le(), None
    if param_id == 57:
        n = data_reader.read_int16_le()
        param_value = Obj()
        param_value.font_list = []
        for _i in range(n):
            font_display_name = data_reader.read_string(data_reader.read_int16_le(), "UTF-8")
            font_name = data_reader.read_string(data_reader.read_int16_le(), "UTF-8")
            param_value.font_list.append([font_display_name, font_name])
        param_value.unk = data_reader.read_int32_le()
        return 'fonts', param_value, None
    if param_id == 64:
        param_value = [(data_reader.read_int32_le(signed=True) / 100) for _i in range(8)]
        return 'quad_verts', param_value, None

    try:
        # these parameters are not used in export process, they are only to keep bits of knowledge of the format and nicer output for format research
        if param_id == 33:
            param_value = data_reader.read_int32_le()
            if param_value != 1:
                logging.warning("parameter 'unit' has non-1 unusual value %s", param_value)
            return 'unit', param_value, None 
        if param_id in [35, 37, 38, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 55, 58, 59, 60, 61, 62]:
            if param_data_size != 4:
                logging.warning("unexpected size=%s of parameter %s", param_data_size, param_id)
            val = data_reader.read_int32_le()
            if param_id == 59: return 'skew_angle_1', val, None  # skew parameters are not used because they are already defined by quad_verts
            if param_id == 60: return 'skew_angle_2', val, None 
            return f'~param_int_{param_id}_unk', val, None
        if param_id == 47:
            data_reader.read_int16_le()
            a, b = data_reader.read_int32_le(), data_reader.read_int32_le()
            if a != 50 or b != 0:
                logging.warning('unexpected values in parsing %s %s', a, b)
            param_value = data_reader.read_string(data_reader.read_int16_le(), "UTF-8")
            return '~param_unk_font', param_value, None
        if param_id  == 39:
            param_value = [ data_reader.read_int32_le(), data_reader.read_int32_le() ]
            return '~param_39_unk_pair', param_value, None
        if param_id == 63:
            param_value = [ data_reader.read_int32_le(), data_reader.read_int32_le() ]
            return 'box_size', param_value, None
    except Exception as e:
        logging.warning("can't parse param_id=%s while parsing text layer: %s", param_id, repr(e))
        traceback.print_exc()

    # '~' in the names to group and sort them to end of output
    return f'~param{param_id}', data_reader.read_n(param_data_size), None

def parse_layer_text_attribute(data):
    data_reader_all_params = DataReader(data)
    clip_text_params = {}

    while data_reader_all_params.ofs < len(data_reader_all_params.data):
        param_id = data_reader_all_params.read_int32_le()
        param_data_size = data_reader_all_params.read_int32_le()
        if param_data_size == 0:
            continue
        param_data = data_reader_all_params.read_n(param_data_size)
        data_reader = DataReader(param_data)

        param_name, param_value, debug_print_extra = parse_layer_text_attribute_param(param_id, data_reader)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('param_name= [%s] param_id=%s extra=%s value=%s', param_name, param_id, debug_print_extra, param_value)
            logging.debug(param_data.hex(' '))

        if param_name == None:
            continue

        if param_name in clip_text_params:
            logging.warning("warning: duplicated parameter '%s'", param_name)
        clip_text_params[param_name] = param_value

        if data_reader.ofs != param_data_size:
            raise ValueError(param_id)

    return clip_text_params

paragraph_template = '''
			<<
				/ParagraphSheet
				<<
					/DefaultStyleSheet 0
					/Properties
					<<
						/Justification {style_paragraph_align}
						/FirstLineIndent 0.0
						/StartIndent 0.0
						/EndIndent 0.0
						/SpaceBefore 0.0
						/SpaceAfter 0.0
						/AutoHyphenate true
						/HyphenatedWordSize 6
						/PreHyphen 2
						/PostHyphen 2
						/ConsecutiveHyphens 8
						/Zone 36.0
						/WordSpacing [ .8 1.0 1.33 ]
						/LetterSpacing [ 0.0 0.0 0.0 ]
						/GlyphSpacing [ 1.0 1.0 1.0 ]
						/AutoLeading 1.2
						/LeadingType 0
						/Hanging false
						/Burasagari false
						/KinsokuOrder 0
						/EveryLineComposer false
					>>
				>>
				/Adjustments
				<<
					/Axis [ 1.0 0.0 1.0 ]
					/XY [ 0.0 0.0 ]
				>>
			>>'''

psd_text_engine_font_template = '''
		<<
			/Name {font_name}
			/Script 0
			/FontType {font_type}
			/Synthetic 0
		>>'''

psd_text_bold = '''
						/FauxBold true'''
psd_text_italic = '''
						/FauxItalic true'''
psd_text_underline = '''
						/Underline true'''
psd_text_strike = '''
						/Strikethrough true'''

psd_text_engine_style_template = '''
			<<
				/StyleSheet
				<<
					/StyleSheetData
					<<
						/Font {style_font_index}
						/FontSize {style_font_size}{style_bold}{style_italic}{style_underline}{style_strike}
						/AutoKerning {style_auto_kerning}
						/Kerning 0
						/Language 14
						/FillColor
						<<
							/Type 1
							/Values [ 1.0 {style_color} ]
						>>
					>>
				>>
			>>'''


psd_text_engine_script = '''

<<
	/EngineDict
	<<
		/Editor
		<<
			/Text {exported_text}
		>>
		/ParagraphRun
		<<
			/DefaultRunData
			<<
				/ParagraphSheet
				<<
					/DefaultStyleSheet 0
					/Properties
					<<
					>>
				>>
				/Adjustments
				<<
					/Axis [ 1.0 0.0 1.0 ]
					/XY [ 0.0 0.0 ]
				>>
			>>
			/RunArray [{paragraph_list}
			]
			/RunLengthArray {paragraph_list_lenghts}
			/IsJoinable 1
		>>
		/StyleRun
		<<
			/DefaultRunData
			<<
				/StyleSheet
				<<
					/StyleSheetData
					<<
					>>
				>>
			>>
			/RunArray [{style_run_array}
			]
			/RunLengthArray [ {style_run_array_lengths} ]
			/IsJoinable 2
		>>
		/GridInfo
		<<
			/GridIsOn false
			/ShowGrid false
			/GridSize 18.0
			/GridLeading 22.0
			/GridColor
			<<
				/Type 1
				/Values [ 0.0 0.0 0.0 1.0 ]
			>>
			/GridLeadingFillColor
			<<
				/Type 1
				/Values [ 0.0 0.0 0.0 1.0 ]
			>>
			/AlignLineHeightToGridFlags false
		>>
		/AntiAlias 4
		/UseFractionalGlyphWidths true
		/Rendered
		<<
			/Version 1
			/Shapes
			<<
				/WritingDirection 0
				/Children [
				<<
					/ShapeType 0
					/Procession 0
					/Lines
					<<
						/WritingDirection 0
						/Children [ ]
					>>
					/Cookie
					<<
						/Photoshop
						<<
							/ShapeType 0
							/PointBase [ 0.0 0.0 ]
							/Base
							<<
								/ShapeType 0
								/TransformPoint0 [ 1.0 0.0 ]
								/TransformPoint1 [ 0.0 1.0 ]
								/TransformPoint2 [ 0.0 0.0 ]
							>>
						>>
					>>
				>>
				]
			>>
		>>
	>>
	/ResourceDict
	<<
		/KinsokuSet [
		<<
			/Name 'PhotoshopKinsokuHard'
			/NoStart '、。，．・：；？！ー―’”）〕］｝〉》」』】ヽヾゝゞ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ゛゜?!)]},.:;℃℉¢％‰'
			/NoEnd '‘“（〔［｛〈《「『【([{￥＄£＠§〒＃'
			/Keep '―‥'
			/Hanging '、。.,'
		>>
		<<
			/Name 'PhotoshopKinsokuSoft'
			/NoStart '、。，．・：；？！’”）〕］｝〉》」』】ヽヾゝゞ々'
			/NoEnd '‘“（〔［｛〈《「『【'
			/Keep '―‥'
			/Hanging '、。.,'
		>>
		]
		/MojiKumiSet [
		<<
			/InternalName 'Photoshop6MojiKumiSet1'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet2'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet3'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet4'
		>>
		]
		/TheNormalStyleSheet 0
		/TheNormalParagraphSheet 0
		/ParagraphSheetSet [
		<<
			/Name 'Normal RGB'
			/DefaultStyleSheet 0
			/Properties
			<<
				/Justification 0
				/FirstLineIndent 0.0
				/StartIndent 0.0
				/EndIndent 0.0
				/SpaceBefore 0.0
				/SpaceAfter 0.0
				/AutoHyphenate true
				/HyphenatedWordSize 6
				/PreHyphen 2
				/PostHyphen 2
				/ConsecutiveHyphens 8
				/Zone 36.0
				/WordSpacing [ .8 1.0 1.33 ]
				/LetterSpacing [ 0.0 0.0 0.0 ]
				/GlyphSpacing [ 1.0 1.0 1.0 ]
				/AutoLeading 1.2
				/LeadingType 0
				/Hanging false
				/Burasagari false
				/KinsokuOrder 0
				/EveryLineComposer false
			>>
		>>
		]
		/StyleSheetSet [
		<<
			/Name 'Normal RGB'
			/StyleSheetData
			<<
				/Font 2
				/FontSize 12.0
				/FauxBold false
				/FauxItalic false
				/AutoLeading true
				/Leading 0.0
				/HorizontalScale 1.0
				/VerticalScale 1.0
				/Tracking 0
				/AutoKerning true
				/Kerning 0
				/BaselineShift 0.0
				/FontCaps 0
				/FontBaseline 0
				/Underline false
				/Strikethrough false
				/Ligatures true
				/DLigatures false
				/BaselineDirection 2
				/Tsume 0.0
				/StyleRunAlignment 2
				/Language 0
				/NoBreak false
				/FillColor
				<<
					/Type 1
					/Values [ 1.0 0.0 0.0 0.0 ]
				>>
				/StrokeColor
				<<
					/Type 1
					/Values [ 1.0 0.0 0.0 0.0 ]
				>>
				/FillFlag true
				/StrokeFlag false
				/FillFirst true
				/YUnderline 1
				/OutlineWidth 1.0
				/CharacterDirection 0
				/HindiNumbers false
				/Kashida 1
				/DiacriticPos 2
			>>
		>>
		]
		/FontSet [{font_set}
		]
		/SuperscriptSize .583
		/SuperscriptPosition .333
		/SubscriptSize .583
		/SubscriptPosition .333
		/SmallCapSize .7
	>>
	/DocumentResources
	<<
		/KinsokuSet [
		<<
			/Name 'PhotoshopKinsokuHard'
			/NoStart '、。，．・：；？！ー―’”）〕］｝〉》」』】ヽヾゝゞ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ゛゜?!)]},.:;℃℉¢％‰'
			/NoEnd '‘“（〔［｛〈《「『【([{￥＄£＠§〒＃'
			/Keep '―‥'
			/Hanging '、。.,'
		>>
		<<
			/Name 'PhotoshopKinsokuSoft'
			/NoStart '、。，．・：；？！’”）〕］｝〉》」』】ヽヾゝゞ々'
			/NoEnd '‘“（〔［｛〈《「『【'
			/Keep '―‥'
			/Hanging '、。.,'
		>>
		]
		/MojiKumiSet [
		<<
			/InternalName 'Photoshop6MojiKumiSet1'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet2'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet3'
		>>
		<<
			/InternalName 'Photoshop6MojiKumiSet4'
		>>
		]
		/TheNormalStyleSheet 0
		/TheNormalParagraphSheet 0
		/ParagraphSheetSet [
		<<
			/Name 'Normal RGB'
			/DefaultStyleSheet 0
			/Properties
			<<
				/Justification 0
				/FirstLineIndent 0.0
				/StartIndent 0.0
				/EndIndent 0.0
				/SpaceBefore 0.0
				/SpaceAfter 0.0
				/AutoHyphenate true
				/HyphenatedWordSize 6
				/PreHyphen 2
				/PostHyphen 2
				/ConsecutiveHyphens 8
				/Zone 36.0
				/WordSpacing [ .8 1.0 1.33 ]
				/LetterSpacing [ 0.0 0.0 0.0 ]
				/GlyphSpacing [ 1.0 1.0 1.0 ]
				/AutoLeading 1.2
				/LeadingType 0
				/Hanging false
				/Burasagari false
				/KinsokuOrder 0
				/EveryLineComposer false
			>>
		>>
		]
		/StyleSheetSet [
		<<
			/Name 'Normal RGB'
			/StyleSheetData
			<<
				/Font 2
				/FontSize 12.0
				/FauxBold false
				/FauxItalic false
				/AutoLeading true
				/Leading 0.0
				/HorizontalScale 1.0
				/VerticalScale 1.0
				/Tracking 0
				/AutoKerning true
				/Kerning 0
				/BaselineShift 0.0
				/FontCaps 0
				/FontBaseline 0
				/Underline false
				/Strikethrough false
				/Ligatures true
				/DLigatures false
				/BaselineDirection 2
				/Tsume 0.0
				/StyleRunAlignment 2
				/Language 0
				/NoBreak false
				/FillColor
				<<
					/Type 1
					/Values [ 1.0 0.0 0.0 0.0 ]
				>>
				/StrokeColor
				<<
					/Type 1
					/Values [ 1.0 0.0 0.0 0.0 ]
				>>
				/FillFlag true
				/StrokeFlag false
				/FillFirst true
				/YUnderline 1
				/OutlineWidth 1.0
				/CharacterDirection 0
				/HindiNumbers false
				/Kashida 1
				/DiacriticPos 2
			>>
		>>
		]
		/FontSet [{font_set}
		]
		/SuperscriptSize .583
		/SuperscriptPosition .333
		/SubscriptSize .583
		/SubscriptPosition .333
		/SmallCapSize .7
	>>
>>'''

# PSD file generation requires knowledge of bold and italic font names when only the base name is
# available. This table provides information on how font names are extended for bold and
# italic styles. Typically, font names are extended with "Bold" or "Italic," either joined
# with a hyphen or directly appended. These variations are listed under the
# "dash" and "no_dash" subsections, respectively. The percent sign (%) is a placeholder
# that is replaced by "Bold" or "Italic" or "BoldItalic" to generate font name variants.
# For less regular variants, all existing style variants (regular, bold, italic, bold italic)
# are kept as tuples and are listed in the "tuple" section.
# The entire table is grouped by the property of the fonts (has_bold, has_italic, has_bold_italic)
# This table is generated by the script parse_adobe_fonts.py
font_name_regular_to_style_font_name = {
(True, True, True) :
    {
       "dash": ['AngsanaNew%', 'AngsanaUPC%', 'Aparajita%', 'Arial%MT', 'ArialNarrow%', 'BookAntiqua%', 'BookmanOldStyle%', 'BrowalliaNew%', 'BrowalliaUPC%', 'Calibri%', 'Cambria%', 'Candara%', 'CenturyGothic%', 'Consolas%', 'Constantia%', 'Corbel%', 'CordiaNew%', 'CordiaUPC%', 'CourierNewPS%MT', 'Georgia%', 'Kokila%', 'Meiryo%', 'MeiryoUI%', 'SegoeUI%', 'TimesNewRomanPS%MT', 'Utsaah%', 'Verdana%'],
       "no_dash": ['DilleniaUPC%', 'EucrosiaUPC%', 'FreesiaUPC%', 'IrisUPC%', 'JasmineUPC%', 'KodchiangUPC%', 'LilyUPC%'],
       "tuple": [
            ('LucidaBright', 'LucidaBright-Demi', 'LucidaBright-Italic', 'LucidaBright-DemiItalic'),
            ('LucidaFax', 'LucidaFax-Demi', 'LucidaFax-Italic', 'LucidaFax-DemiItalic'),
            ('PalatinoLinotype-Roman', 'PalatinoLinotype-Bold', 'PalatinoLinotype-Italic', 'PalatinoLinotype-BoldItalic'),
            ('TrebuchetMS', 'TrebuchetMS-Bold', 'TrebuchetMS-Italic', 'Trebuchet-BoldItalic'),
    ]},
(True, False, False) :
    {
       "dash": ['ComicSansMS%', 'David%', 'DejaVuSans%', 'Ebrima%', 'Gautami%', 'Gisha%', 'IskoolaPota%', 'Kalinga%', 'Kartika%', 'KhmerUI%', 'LaoUI%', 'Latha%', 'Leelawadee%', 'LevenimMT%', 'Mangal%', 'MicrosoftNewTaiLue%', 'MicrosoftPhagsPa%', 'MicrosoftTaiLe%', 'MicrosoftYaHei%', 'SegoePrint%', 'SegoeScript%', 'ShonarBangla%', 'Shruti%', 'SimplifiedArabic%', 'Tahoma%', 'TraditionalArabic%', 'Tunga%', 'Vani%', 'Vijaya%', 'Vrinda%'],
       "no_dash": ['FreeSans%', 'SakkalMajalla%'],
       "tuple": [ ('BerlinSansFB-Reg', 'BerlinSansFB-Bold') , ('MalgunGothicRegular', 'MalgunGothicBold') , ('MicrosoftJhengHeiRegular', 'MicrosoftJhengHeiBold'), 
    ]},
(True, True, False) : { "dash": ['Garamond%'] , "no_dash": ['BellMT%'] , "tuple": [ ('CalifornianFB-Reg', 'CalifornianFB-Bold', 'CalifornianFB-Italic') , ]},
(False, True, False) : { "dash": [] , "no_dash": ['FranklinGothic-Medium%'] , "tuple": [ ('HighTowerText-Reg', 'HighTowerText-Italic') , ]},
}

# Mapping font_name -> tuple (font_name_for_bold, font_name_for_italic, font_name_for_bold_italic). font names in the tuple can be None.
table_regular_font_to_styled = {}
def init_table_regular_font_to_styled():
    # font_name_regular_to_style_font_name -> table_regular_font_to_styled
    # Expand compact table font_name_regular_to_style_font_name to more
    # convenient access: just mapping font name to font styles
    for style_flags, style_fonts in font_name_regular_to_style_font_name.items():
        for dash, field_name in (('-', "dash"), ('', "no_dash")):
            for font_name_template in style_fonts[field_name]:
                basic_font_name = font_name_template.replace('%', '')
                font_style_names = []
                for flag, style_font_name_part in zip(style_flags, ('Bold', 'Italic', 'BoldItalic')):
                    style_font_name_part = dash + style_font_name_part
                    font_style_names.append(font_name_template.replace('%', style_font_name_part) if flag else None)
                table_regular_font_to_styled[basic_font_name] = font_style_names

        for font_name_template in style_fonts["tuple"]:
            true_flag_indices = [i for i, flag in enumerate(style_flags) if flag]
            assert len(true_flag_indices) + 1 == len(font_name_template)
            font_style_names = [None]*len(style_flags)
            for true_flag_index, font_name in zip(true_flag_indices, font_name_template[1:]):
                font_style_names[true_flag_index] = font_name
            basic_font_name = font_name_template[0]
            table_regular_font_to_styled[basic_font_name] = font_style_names

init_table_regular_font_to_styled()


# maps x to y for bezier curve created by p0, p1, p2 2d-points
def bezier_interpolation_3point(p0, p1, p2, x):
    # Coefficients for the quadratic Bezier curve for mapping t to x.
    # multiplicator '2' for b is implicit, to use simplified formula for quadratic equation at^2 + 2bt + t
    a = p0[0] - 2 * p1[0] + p2[0]
    b = p1[0] - p0[0]
    c = p0[0]

    # Function to calculate the corresponding t for a given x
    def solve_for_t(x):
        discriminant = b**2 - a * (c - x)
        if discriminant < 0:
            raise ValueError("The x value is not in the range of the Bezier curve.")

        return (c - x) / (-b - math.sqrt(discriminant)) # Citardauq modification of quadratic roots formula, to handle edge case with a = 0

    # Function to calculate y for a given t using the Bezier formula
    def calculate_y(t):
        return (1-t)**2 * p0[1] + 2 * (1-t)*t * p1[1] + t**2 * p2[1]

    return calculate_y(solve_for_t(x))
            
def fix_unexpected_exported_array(points):
    if not points:
        logging.warning("empty list of curve mapping %s", repr(points))
        points.append((0, 0))
        points.append((65535, 65535))

    points.sort(key = lambda p: p[0])
    # we don't need to handle X less than 0 or larger than 65535, because they are exported from 2-bytes. But exported array can miss start 0 and end 65535 
    if points[0][0] != 0:
        logging.warning("first point x is not 0 in curve %s", repr(points))
        points.insert(0, (0, 0))

    if points[-1][0] != 65535:
        logging.warning("last point x is not 65535 in curve %s", repr(points))
        points.append((65535, 65535))

    assert len(points) >= 2

def linear_interpolation_for_clip_curve_filter(points):
    assert len(points) == 2
    assert points[0][0] == 0
    assert points[1][0] == 65535
    y0, y1 = points[0][1], points[1][1] 
    result = [None] * 256
    for x_i in range(256):
        x = 257*x_i # last x_i=0xff maps exactly to 0xffff by this multiplication
        assert 0 <= x <= 65535
        y = y0 + x * (y1 - y0) / 65535
        assert 0 <= y <= 65535, y
        result[x_i] = int(y + 0.5) // 256
        assert 0 <= result[x_i] <= 255, result[x_i]
    return result


def multi_bezier_interpolation_for_clip_curve_filter(points):
    assert len(points) >= 2, points 
    if len(points) == 2:
        # mapping type of one channel force to create table mappings for all other channels, even 2-points linear ones
        return linear_interpolation_for_clip_curve_filter(points)

    midpoints = [tuple(0.5 * (points[i][j] + points[i+1][j]) for j in (0, 1)) for i in range(1, len(points) - 2)] 
    endpoints = [points[0]] + midpoints + [points[-1]] # start/end of independent interpolation regions

    n = len(endpoints)
    assert n == len(points) - 1 , (points, endpoints)
    assert 0 == endpoints[0][0]
    assert 65535 == endpoints[-1][0]
    assert all(endpoints[i][0] <= endpoints[i+1][0] for i in range(n-1)), (points)

    result = [None] * 256

    def is_x_inside(x, i_seg):
        b = endpoints[i_seg+1][0]
        # last interval is closed [a,b], other are half-opened [a,b)
        return (endpoints[i_seg][0] <= x) and (x <= b if (i_seg == n-2) else x < b) 

    i_seg = 0
    for x_i in range(256):
        x = 257*x_i # last x_i=0xff maps exactly to 0xffff by this multiplication
        assert 0 <= x <= 65535

        while not is_x_inside(x, i_seg):
            i_seg += 1
            assert i_seg < n-1

        start = endpoints[i_seg]
        middle = points[i_seg+1]
        end = endpoints[i_seg+1]

        # comparison with ends of interval to avoid 0/0 division for vertical segments. 
        # values of x and input points are integers, so there no problem with
        # "almost equal" values for inexact floats
        if x == start[0]:
            y = start[1]
        elif x == end[0]:
            y = end[1]
        else:
            y = bezier_interpolation_3point(start, middle, end, x)
        result[x_i] = int(y + 0.5) // 256
        assert 0 <= result[x_i] <= 255, result[x_i]

    return result


GradientColorStop = namedtuple('GradientColorStop', ['r', 'g', 'b', 'opacity', 'position'])

GradientGeometryData = namedtuple('GradientGeometryData', [
    'repeat_mode', 'shape', 'anti_aliasing',
    'ellipse_second_diameter',
    'pos_start_x', 'pos_start_y', 'pos_end_x', 'pos_end_y'
])

def parse_gradient_color_stop(f):
    r = read_csp_int(f) >> 24
    g = read_csp_int(f) >> 24
    b = read_csp_int(f) >> 24
    opacity = read_csp_int(f) >> 24
    _is_current_color = read_csp_int(f) # 0 - use color from gradient definition. 1/2 - color is taken from current editor color/secondary color.
    position = read_csp_int(f) * 100 // (2**15)  # 0 .. 100
    num_curve_points = read_csp_int(f)

    if num_curve_points != 0:
        logging.warning("Found reference to curve in gradient definition. Export of curves in gradients is not supported.")

    # curve points data follows in this section, 16 bytes per each point, but this script can't export them to PSD anyway.

    return GradientColorStop(r, g, b, opacity, position)

def parse_gradient_flat_color_data(f):
    is_flat = bool(read_csp_int(f))
    if not is_flat:
        return (False, None)
    flat_rgb_tuple = tuple(read_csp_int(f) >> 24 for _ in range(3))
    return (True, flat_rgb_tuple)

def parse_gradient_geometry_data(f):
    repeat_mode = read_csp_int(f) # repeat mode outside gradient boundaries.  clip/repeat/mirror repeat/empty - 0,1,2,3
    shape = read_csp_int(f) # linear/circle/ellipse - 0,1,2
    anti_aliasing = read_csp_int(f) # 0/1
    _diameter_unused = read_csp_double(f) # always 2x of center and endpoint distance for circle/ellipse, have no meaning for linear
    ellipse_second_diameter = read_csp_double(f)
    _angle_rotation_unused = read_csp_double(f) # angle of rotation in degrees, but it's updated only for circle/ellipse.
    pos_start_x = read_csp_double(f)
    pos_start_y = read_csp_double(f)
    pos_end_x = read_csp_double(f)
    pos_end_y = read_csp_double(f)
    return GradientGeometryData(repeat_mode, shape, anti_aliasing, ellipse_second_diameter, pos_start_x, pos_start_y, pos_end_x, pos_end_y)

def parse_gradient_colors(f):
    _unknown = read_csp_int(f)
    _unknown = read_csp_int(f)
    num_color_stops = read_csp_int(f)
    _unknown = read_csp_int(f)

    color_stops = tuple(parse_gradient_color_stop(f) for _ in range(num_color_stops))
    return color_stops


def parse_gradation_fill_data_of_gradient_layers(data):
    f = io.BytesIO(data)
    _size_of_data_structure = read_csp_int(f)
    _unknown = read_csp_int(f)

    color_stops = None
    geometry_data = None
    flat_color_data = None

    while (param_name := read_csp_unicode_str(f)) != None:
        if param_name == "GradationData":
            section_size = read_csp_int(f)
            color_stops = parse_gradient_colors(io.BytesIO(f.read(section_size)))
        elif param_name == "GradationSettingAdd0001":
            section_size = read_csp_int(f)
            flat_color_data = parse_gradient_flat_color_data(io.BytesIO(f.read(section_size)))
        elif param_name == "GradationSetting":
            geometry_data = parse_gradient_geometry_data(f)
        else:
            already_has_information = flat_color_data and (flat_color_data[0] or (color_stops and geometry_data)) #pylint: disable=unsubscriptable-object
            if already_has_information:
                break
            logging.warning("unknown parameter %s, trying to skip it as typical parameter", repr(param_name))
            section_size = read_csp_int(f)
            _unused = f.read(section_size)

    if flat_color_data and flat_color_data[0]:
        return flat_color_data # returns (True, (r,g,b)) for flat color
    else:
        if not (color_stops and geometry_data):
            color_stops_length = None if color_stops==None else len(color_stops)
            logging.error(
                "could not find color stops information in gradient definition, color_stops=%s, color_stops_length=%s, geometry_data=%s, [%s]", 
                bool(color_stops), color_stops_length, bool(geometry_data), repr(data)[0:10000])
            return None
        return (False, (color_stops, geometry_data))

def save_preview_image(output_filename, sqlite_info):
    from PIL import Image
    if not sqlite_info.canvas_preview_data and len(sqlite_info.canvas_preview_data):
        logging.error("canvas preview data not found in .clip file")
        sys.exit(1)

    canvas_width = int(sqlite_info.width)
    canvas_height = int(sqlite_info.height)

    preview = Image.open(io.BytesIO(sqlite_info.canvas_preview_data[0]))
    logging.info("saved image preview size= %s x %s px; %s; canvas size = %s x %s", preview.size[0], preview.size[1], preview.mode, canvas_width, canvas_height)
    preview.save(output_filename)

def save_psd(output_psd, chunks, sqlite_info, layer_ordered):
    psd_version = cmd_args.psd_version 
    layer_bitmaps = get_layers_bitmaps(chunks, sqlite_info)

    def write_int_n(f, i, size, signed):
        if signed:
            k = (1 << (8*size - 1))
            assert -k <= i < k, (i, size, signed)
        else:
            assert 0 <= i < (1 << 8*size), (i, size)
        f.write(i.to_bytes(size, 'big', signed=signed))

    def write_int(f, i): write_int_n(f, i, 4, False)
    def write_int_signed(f, i): write_int_n(f, i, 4, True)
    def write_int2(f, i): write_int_n(f, i, 2, False)
    def write_int1(f, i): write_int_n(f, i, 1, False)

    def write_int_psb(f, i):
        assert 0 <= i < 2**64, i
        if psd_version == 1:
            assert 0 <= i < 2**32, (i, "size is too large for 32-bit psd, try Big Psd output (PSB) with --psd-version=2")
        f.write(i.to_bytes(4*psd_version, 'big'))

    def check_pil_import():
        try:
            from PIL import Image
        except ImportError:
            logging.error("Error: Failed to import PIL. Ensure it's installed (search for python 'pillow' package) or explicitly use --blank-psd-preview to avoid import.")
            raise

    def export_canvas_preview(f):
        logging.info("exporting canvas preview")
        # attempt to export root folder bitmap failed - ClipStudioPaint often supply partially saved bitmaps for folders with holes at random places.
        # so CanvasPreview is used (but it's downscaled). It's not original resolution quality image, but at least can be used for thumbnails.
        canvas_width = int(sqlite_info.width)
        canvas_height = int(sqlite_info.height)
        buf = bytearray(canvas_width*2)
        if cmd_args.blank_psd_preview:
            channels = [ bytes(canvas_width*canvas_height) for _ch in range(3) ]
        else:
            check_pil_import()
            from PIL import Image
            if len(sqlite_info.canvas_preview_data):
                preview = Image.open(io.BytesIO(sqlite_info.canvas_preview_data[0]))
                w, h = preview.size
                logging.info('%s', f"preview size {w=} {h=}; {canvas_width=} {canvas_height=}")
                img = preview.resize((canvas_width, canvas_height), Image.NEAREST) # NEAREST is more friendly to RLE compression, and image is downscaled to thumbnail in previews anyway.
                if img.mode != "RGB":
                    img = img.convert("RGB")
                assert img.size == (canvas_width, canvas_height)
            else:
                img = Image.new("RGB", (canvas_width, canvas_height), (128,128,128))
            channels = [ ch.tobytes() for ch in img.split() ]
        rle_lines = []
        for channel_pixel_data in channels:
            for i in range(canvas_height):
                rle_len = rle_compress( channel_pixel_data[i*canvas_width:(i+1)*canvas_width], canvas_width, 128, buf)
                rle_lines.append([buf[:rle_len]])
        channel_output_tmp_buf = bytearray(2 + 3*canvas_height*(canvas_width + 4))
        canvas_preview_rle = join_rle_scanlines_to_psd_channel(rle_lines, channel_output_tmp_buf, psd_version)
        f.write(canvas_preview_rle)

    def write_layer_tags(f, layer_tags):
        tags_for_8_byte_length_size = set(b'LMsk Lr16 Lr32 Layr Mt16 Mt32 Mtrn Alph FMsk lnk2 FEid FXid PxSD'.split())
        for tag_name, data in layer_tags:
            assert isinstance(tag_name, bytes), repr(tag_name)
            long_size = (tag_name in tags_for_8_byte_length_size) and psd_version == 2
            f.write(b'8BIM')
            f.write(tag_name)
            size_of_size = (8 if long_size else 4)
            write_int_n(f, len(data), size=size_of_size, signed=False)
            f.write(data)

    def write_bitmap_and_mask_channel_info_and_get_binary_data(f, layer_type, layer):
        has_mask = bool(layer and layer_bitmaps[layer.MainId].LayerMaskBitmap)

        def write_channel_info(layer_channels, channel_scanlines):
            for psd_channel_tag, channel_data in channel_scanlines:
                write_int2(f, psd_channel_tag % (2**16))
                write_int_psb(f, len(channel_data))
                layer_channels.append(channel_data)

        layer_channels = []
        filter_layer_info = getattr(layer, 'FilterLayerInfo', None)
        has_fill_color = getattr(layer, 'DrawColorEnable', None)

        layer_bitmap_info_for_export = None
        if layer_type == "lt_bitmap" and not filter_layer_info and not has_fill_color:
            layer_all_bitmaps_info = layer_bitmaps.get(layer.MainId)
            if layer_all_bitmaps_info:
                layer_bitmap_info_for_export = layer_all_bitmaps_info.LayerBitmap
                if not layer_bitmap_info_for_export:
                    logging.warning("layer '%s' has no pixel data bitmap info", layer.LayerName)
            else:
                logging.warning("layer '%s' has no bitmaps info", layer.LayerName)

        if layer_bitmap_info_for_export:
            bitmap_blocks, offscreen_attribute = layer_bitmaps[layer.MainId].LayerBitmap
            channel_scanlines, offset_x, offset_y, bitmap_width, bitmap_height, _default_color = decode_to_psd_rle(offscreen_attribute, bitmap_blocks, psd_version, "layer")
            assert len(channel_scanlines) == 4
            offset_x += layer.LayerOffsetX + layer.LayerRenderOffscrOffsetX
            offset_y += layer.LayerOffsetY + layer.LayerRenderOffscrOffsetY

            #write rectangle: top, left, bottom, right
            write_int_signed(f, offset_y)
            write_int_signed(f, offset_x)
            write_int_signed(f, offset_y + bitmap_height)
            write_int_signed(f, offset_x + bitmap_width)
        else:
            # Clip Studio folder layers have a bitmap, but it's not needed for PSD export 
            empty = b'\0\0'
            channel_scanlines = [ (-1, empty), (0, empty),  (1, empty),  (2, empty) ]
            for _i in range(4):
                write_int(f, 0)

        channel_count = 4 + has_mask
        write_int2(f, channel_count)
        write_channel_info(layer_channels, channel_scanlines)

        mask_data = b''
        if has_mask:
            # add channel information about mask
            bitmap_blocks, offscreen_attribute = layer_bitmaps[layer.MainId].LayerMaskBitmap
            channel_scanlines, offset_x_mask, offset_y_mask, bitmap_width_mask, bitmap_height_mask, default_mask_color = decode_to_psd_rle(offscreen_attribute, bitmap_blocks, psd_version, "mask")

            logging.debug('mask x offsets %s %s %s %s %s', offset_x_mask, layer.LayerMaskOffsetX, layer.LayerMaskOffscrOffsetX, layer.LayerOffsetX, layer.LayerRenderOffscrOffsetX)
            logging.debug('mask y offsets %s %s %s %s %s', offset_y_mask, layer.LayerMaskOffsetY, layer.LayerMaskOffscrOffsetY, layer.LayerOffsetY, layer.LayerRenderOffscrOffsetY)

            offset_x_mask += layer.LayerMaskOffsetX + layer.LayerOffsetX + layer.LayerMaskOffscrOffsetX
            offset_y_mask += layer.LayerMaskOffsetY + layer.LayerOffsetY + layer.LayerMaskOffscrOffsetY


            assert len(channel_scanlines) == 1
            psd_channel_tag = channel_scanlines[0][0]
            assert psd_channel_tag == -2
            write_channel_info(layer_channels, channel_scanlines)

            # produce mask binary data
            f_mask = io.BytesIO()
            write_int_signed(f_mask, offset_y_mask)
            write_int_signed(f_mask, offset_x_mask)
            write_int_signed(f_mask, offset_y_mask + bitmap_height_mask)
            write_int_signed(f_mask, offset_x_mask + bitmap_width_mask)
            write_int1(f_mask, default_mask_color)
            mask_disabled = not bool(layer.LayerVisibility & 2)
            mask_relative_pose = not bool(layer.LayerSelect & 256)
            mask_flags = 2*mask_disabled + mask_relative_pose
            write_int1(f_mask, mask_flags)
            write_int2(f_mask, 0) # padding
            mask_data = f_mask.getvalue()

        return mask_data, layer_channels

    def write_layer_name(f, layer_name):
        # painting programs use default windows/mac local encoding when write this field.
        # Anyway, most of painting programs use 'luni' layer tag with unicode name when read layer name,
        # so it's not super-important to guess encoding.
        layer_name_ascii = escape_bytes_str(layer_name.encode('UTF-8'))[0:255]
        layer_name_len = len(layer_name_ascii)
        write_int1(f, layer_name_len)
        f.write(layer_name_ascii)
        f.write(b'\0' * (-(layer_name_len+1) & 3) ) # padding 4

    def add_fill_color_for_layer(layer_tags, fill_color):
        obj = PsdObjDescriptorWriter(io.BytesIO())
        write_int(obj.f, 16) # descriptor version
        obj.write_obj_header('null', 1)
        obj.write_field('Clr ', 'Objc')
        obj.write_obj_header('RGBC', 3)
        obj.write_doub('Rd  ', fill_color[0])
        obj.write_doub('Grn ', fill_color[1])
        obj.write_doub('Bl  ', fill_color[2])
        layer_tags.append((b'SoCo', obj.f.getvalue()))

    def add_fill_color_for_background_layer(layer_tags, layer):
        has_fill_color = getattr(layer, 'DrawColorEnable', None)
        if has_fill_color:
            logging.info("exporting solid color layer '%s'", layer.LayerName)
            fill_color = [ getattr(layer, 'DrawColorMain' + x, 0)/(2**32 - 1)*255 for x in ['Red', 'Green', 'Blue'] ]
            add_fill_color_for_layer(layer_tags, fill_color)

    def add_layer_color_property(layer_tags, layer):
        use_color = getattr(layer, 'LayerUsePaletteColor', 0)
        fill_color_csp = [ getattr(layer, 'LayerPalette' + x, None) for x in ['Red', 'Green', 'Blue'] ]

        if not use_color or any(x == None for x in fill_color_csp):
            return
        fill_color = [ x/(2**32 - 1)*255 for x in  fill_color_csp]
        # PSD has limited set of colors for layers, CSP allows arbitrary colors. Map the arbitrary color to the closest PSD allowed color.
        colors = [
            (255, 0, 0),     # Red
            (255, 128, 0),   # Orange
            (255, 255, 0),   # Yellow
            (128, 255, 64),  #  Green
            (128, 128, 255), #  Blue
            (200, 128, 255), #  Purple
            (100, 100, 100), #  Gray
        ]
        def distance(color1, color2):
            return sum((x-y)**2 for x, y in zip(color1, color2))

        _, closest_color_index = min( (distance(c, fill_color), i) for i, c in enumerate(colors) )
        closest_color_index += 1
        layer_tags.append((b'lclr', closest_color_index.to_bytes(2, 'big') + b'\x00' * 6))


    def add_stroke_outline(layer_tags, layer):
        layer_effect_data = getattr(layer, 'LayerEffectInfo', None)
        if not layer_effect_data:
            return

        param_name = 'EffectEdge'
        param_name_search = len(param_name).to_bytes(4, 'big') + param_name.encode('UTF-16BE')
        i = layer_effect_data.find(param_name_search) # allow parameter to be anywhere in data
        if i == -1:
            logging.warning("can't find EffectEdge parameter in LayerEffectInfo data %s", repr(layer_effect_data)[0:10000])
            return
        f = io.BytesIO(layer_effect_data[i+len(param_name_search):])
        enabled = read_csp_int(f)
        thickness = read_csp_double(f)
        rgb = [read_csp_int(f) >> 24 for _ in range(3)]
        logging.debug('EffectEdge %s %s %s', enabled, thickness, rgb)

        if not enabled:
            return

        f = io.BytesIO()
        write_int(f, 0)
        write_int(f, 16) # descriptor version
        obj = PsdObjDescriptorWriter(f)

        obj.write_obj_header('null', 3)
        obj.write_untf('Scl ', '#Prc', 100.0)
        obj.write_bool('masterFXSwitch', True)

        # Third item (object)
        obj.write_field('FrFX', 'Objc')
        obj.write_obj_header('FrFX', 7)

        obj.write_bool('enab', True)
        obj.write_enum2('Styl', 'FStl', 'OutF')
        obj.write_enum2('PntT', 'FrFl', 'SClr')
        obj.write_enum2('Md  ', 'BlnM', 'Nrml')
        obj.write_untf('Opct', '#Prc', 100.0)
        obj.write_untf('Sz  ', '#Pxl', thickness + 0.5)

        # color, 7-th element in FrFx
        obj.write_field('Clr ', 'Objc')
        obj.write_obj_header('RGBC', 3)
        obj.write_doub('Rd  ', rgb[0])
        obj.write_doub('Grn ', rgb[1])
        obj.write_doub('Bl  ', rgb[2])

        data = obj.f.getvalue()
        data += bytes(-len(data) % 4) # padding 4 bytes
        layer_tags.append((b'lfx2', data))



    def add_filter_layer_levels_info(layer_tags, d):
        def map_csp_middle_to_psd_inv_gamma_for_levels(x):
            if x <= 0:
                return 100 # set default 1.0 for invalid inputs
            inv_gamma = -math.log2(x)
            inv_gamma = max(0.1, min(10.26, inv_gamma)) # clip studio paint clipping
            inv_gamma = max(1, min(999, int(0.5 + inv_gamma * 100))) # photoshop parameter restrictions
            return inv_gamma

        psd_default_lvls_entry = [ 0x0000, 0x00ff,    0x0000, 0x00ff, 100 ]
        psd_lvls_array = [ list(psd_default_lvls_entry) for _ in range(29) ]
        i_level_entry = 0
        while d.left():
            clip_bottom, middle, clip_top, black_map, white_map = [d.read_int16_be() for _ in range(5)]
            if i_level_entry < 4: # first four entries are for common levels, r, g, b and other are uknown 
                middle_ratio = (middle-clip_bottom)  / max(1, clip_top - clip_bottom)
                inv_gamma_psd = map_csp_middle_to_psd_inv_gamma_for_levels(middle_ratio)
                psd_lvls_array[i_level_entry][0] = clip_bottom // 256
                psd_lvls_array[i_level_entry][1] = clip_top // 256
                psd_lvls_array[i_level_entry][2] = black_map // 256
                psd_lvls_array[i_level_entry][3] = white_map // 256
                psd_lvls_array[i_level_entry][4] = inv_gamma_psd
                input_floor, input_ceil, output_floor, output_ceil, inversed_gamma = psd_lvls_array[i_level_entry]
                logging.debug("Levels adjustment [%s] -> [%s]", f'csp: {clip_bottom=}, {middle=}, {clip_top=}, {black_map=}, {white_map=}', f'psd: {input_floor=} {input_ceil=} {output_floor=} {output_ceil=} {inversed_gamma=}') 
                i_level_entry += 1
        result = io.BytesIO()
        write_int2(result, 2) # version
        for e in psd_lvls_array:
            for v in e:
                write_int2(result, v)
        result.write(b'Lvls')
        write_int2(result, 3) # version
        write_int2(result, 33 + 29) # write 33 default entries to match exactly the PSD example
        for _i in range(33):
            for v in psd_default_lvls_entry:
                write_int2(result, v)
        write_int2(result, 0) # don't known, maybe padding
        layer_tags.append((b'levl', result.getvalue()))


    def add_filter_layer_curve_info(layer_tags, d):
        array_of_point_arrays = []
        arrays_start_offset = d.ofs
        for i in range(32):
            d.ofs = arrays_start_offset + i * 130
            points = [(d.read_int16_be(), d.read_int16_be()) for _ in range(d.read_int16_be())]
            fix_unexpected_exported_array(points)
            array_of_point_arrays.append(points)

        if all(len(points) == 2 for points in array_of_point_arrays):
            is_mapping = False
            array_of_point_arrays_psd = [[(y//256, x//256) for x, y in points] for points in array_of_point_arrays]
        else:
            is_mapping = True
            array_of_point_arrays_psd = [multi_bezier_interpolation_for_clip_curve_filter(points) for points in array_of_point_arrays]

        DEFAULT_POINT_SET = [(0, 0), (255, 255)]
        DEFAULT_POINT_TABLE = list(range(256))

        def is_default(points):
            if is_mapping:
                return (points == DEFAULT_POINT_TABLE)
            return (points == DEFAULT_POINT_SET)

        default_is_printed = False
        assert len(array_of_point_arrays) == len(array_of_point_arrays_psd)
        for i, (points_orig, points_psd) in enumerate(zip(array_of_point_arrays, array_of_point_arrays_psd)):
            this_is_default_points = (points_orig == [(0, 0), (65535, 65535)]) and is_default(points_psd)
            do_print = False
            if this_is_default_points:
                if not default_is_printed:
                    logging.debug("channel %s, default values, default channel values are printed only once", i)
                    default_is_printed = True
                    do_print = True
            else:
                do_print = True
            if do_print:
                logging.debug("channel %s, input curve points %s", i, points_orig)
                logging.debug("channel %s, output curve points %s", i, points_psd)

        f = io.BytesIO()

        write_int1(f, int(is_mapping))
        write_int2(f, 1) # version

        channel_bit_set = 0
        channel_count = 0

        for i, points in enumerate(array_of_point_arrays_psd):
            if not is_default(points):
                channel_count += 1
                channel_bit_set |= (1 << i)

        def write_points(write_channel_id):
            for channel_id, points in enumerate(array_of_point_arrays_psd):
                if not is_default(points):
                    if write_channel_id:
                        write_int2(f, channel_id)
                    if is_mapping:
                        for y in points:
                            assert 0 <= y < 256 and int(y) == y, y
                            write_int1(f, y)
                    else:
                        write_int2(f, len(points))
                        for x, y in points:
                            assert 0 <= x < 256 and int(x) == x, x
                            assert 0 <= y < 256 and int(y) == y, y
                            write_int2(f, x)
                            write_int2(f, y)

        write_int(f, channel_bit_set)
        write_points(False)

        f.write(b'Crv ')
        
        write_int2(f, 3 if is_mapping else 4) # extra version
        write_int(f, channel_count)
        write_points(True)

        f.write(bytes(-f.tell() % 4)) # padding
        layer_tags.append((b'curv', f.getvalue()))

    def add_filter_layer_hue_saturation(layer_tags, d):
        hue, saturation, lightness = [d.read_int32_be(signed = True) for _ in range(3)]

        logging.debug('%s', f'exporting {hue=} {saturation=}, {lightness=}')
        logging.warning('HSV filter in PSD works not the same way as in Clip Studio Paint. Review output PSD and adjust values to match original.')

        # Other bytes are related to "colorization" feature or HSV settings 
        # relevant to specific hexants in photoshop UI dialogue
        # Seems, no need to write these non-used values in structured way.
        data = (
            # version / boolean False with meaning "use HSV, not colorization" / padding / colorization settings not used because of flag
            bytes.fromhex('0002' + '00' + '00' + 'ffce00190000') + 
            b''.join([x.to_bytes(2, 'big', signed=True) for x in [hue, saturation, lightness]]) +
            # 6 records of 4 + 3 short ints, not used in HSV when 'Master' mode in UI is used.
            bytes.fromhex(
                '013b0159000f002d' + '000000000000' + 
                '000f002d004b0069' + '000000000000' + 
                '004b0069008700a5' + '000000000000' + 
                '008700a500c300e1' + '000000000000' +
                '00c300e100ff011d' + '000000000000' + 
                '00ff011d013b0159' + '000000000000' 
            )
        )
        layer_tags.append((b'hue2', data))

    def add_filter_layer_info(layer_tags, layer):
        filter_info = getattr(layer, 'FilterLayerInfo', None)
        if filter_info:
            d = DataReader(filter_info)
            filter_index = d.read_int32_be()
            #print(f'{layer.LayerName}:', filter_index)
            input_data_size = d.read_int32_be()
            if input_data_size != d.left():
                logging.warning("filter data size does not match it's own size field (%s != %s)", input_data_size, d.left())
            filter_name_table = { 1:"brit", 2:"levl", 3:"curv", 4:"hue2", 5:"blnc",6:"nvrt",9:"grdm" }
            if filter_index == 1: # Brightness/Contrast
                brightness = d.read_int32_be()
                contrast = d.read_int32_be()
                layer_tags.append((b'brit', bytes(8)))
                obj = PsdObjDescriptorWriter(io.BytesIO())
                write_int(obj.f, 16) # descriptor version
                obj.write_obj_header('null', 7)
                obj.write_long('Vrsn', 1)
                obj.write_long('Brgh', brightness)
                obj.write_long('Cntr', contrast)
                obj.write_long('means', 127, write4as0=False)
                obj.write_bool('Lab ', False)
                obj.write_bool('useLegacy', True)
                obj.write_bool('Auto', False)
                layer_tags.append((b'CgEd', obj.f.getvalue()))
            elif filter_index == 2: # Levels
                add_filter_layer_levels_info(layer_tags, d)
            elif filter_index == 3: # Curves
                add_filter_layer_curve_info(layer_tags, d)
            elif filter_index == 4: # HSV Hue/Saturation
                add_filter_layer_hue_saturation(layer_tags, d)
            else:
                logging.warning("not implemented export of filter type %s %s", filter_index, filter_name_table.get(filter_index, ''))

    def get_gradient_layer_binary_data_for_psd(psd_gradient_offset, angle_psd, scale_psd, psd_gradient_type, color_stops_psd, transparency_stops_psd):
        f = io.BytesIO()
        write_int(f, 16) # descriptor version
        obj = PsdObjDescriptorWriter(f)

        # Writing the main object
        obj.write_obj_header('null', 6)
        obj.write_untf('Angl', '#Ang', angle_psd)
        obj.write_enum2('Type', 'GrdT', psd_gradient_type) # 'Lnr ' / 'Rdl '
        obj.write_bool('Algn', False)
        # key_name=b'Angl' type_name=b'UntF' unit_name=b'#Ang' value=51.71
        # key_name=b'Scl ' type_name=b'UntF' unit_name=b'#Prc' value=90.
        obj.write_untf('Scl ', '#Prc', scale_psd)

        # Writing nested object 'Ofst'
        obj.write_field('Ofst', 'Objc')
        obj.write_obj_header('Pnt ', 2)
        obj.write_untf('Hrzn', '#Prc', psd_gradient_offset[0])
        obj.write_untf('Vrtc', '#Prc', psd_gradient_offset[1])

        # Writing nested object 'Grad'
        obj.write_field('Grad', 'Objc')
        obj.write_obj_header_named('Grdn', 5, 'Gradient')
        obj.write_field('Nm  ', 'TEXT')
        obj.write_unicode_str('$$$/ImportedFromCSP/SomeGradient=Some_Gradient')
        obj.write_enum2('GrdF', 'GrdF', 'CstS')
        obj.write_doub('Intr', 4096.0)

        # Writing list of color stops.
        # (r, g, b, position 0..100 -> 0..4096, middle 0..100)
        obj.write_field('Clrs', 'VlLs')
        write_int(f, len(color_stops_psd))  # list count

        for (r, g, b, position, middle_point) in color_stops_psd:
            obj.write_list_item('Objc')
            obj.write_obj_header('Clrt', 4)
            obj.write_field('Clr ', 'Objc')
            obj.write_obj_header('RGBC', 3)
            obj.write_doub('Rd  ', r)
            obj.write_doub('Grn ', g)
            obj.write_doub('Bl  ', b)
            obj.write_enum2('Type', 'Clry',  'UsrS')
            obj.write_long('Lctn', position)
            obj.write_long('Mdpn', middle_point)

        # Writing list of transparency_stops.
        obj.write_field('Trns', 'VlLs')
        write_int(f, len(transparency_stops_psd))  # list count

        for (opacity, position, middle_point) in transparency_stops_psd:
            obj.write_list_item('Objc')
            obj.write_obj_header('TrnS', 3)
            obj.write_untf('Opct', '#Prc', opacity)
            obj.write_long('Lctn', position)
            obj.write_long('Mdpn', middle_point)

        data = f.getvalue()
        data += bytes(-len(data) % 4) # padding 4 bytes
        return data


    def convert_clip_gradient_to_psd_gradient_math(geometry_data, layer):
        gd = geometry_data
        start_x = gd.pos_start_x + layer.LayerOffsetX  # -LayerRenderOffscrOffsetX - ?
        start_y = gd.pos_start_y + layer.LayerOffsetY
        end_x = gd.pos_end_x + layer.LayerOffsetX
        end_y = gd.pos_end_y + layer.LayerOffsetY
        dx = end_x - start_x
        dy = end_y - start_y
        w, h = int(sqlite_info.width), int(sqlite_info.height)
        shape_name = { 0:"linear", 1:"circular", 2:"ellipse" }.get(gd.shape, "unknown_" + str(gd.shape))
        scale_psd = abs(dy)/h if (abs(dx*h) < abs(dy*w)) else abs(dx)/w
        scale_psd *= 100.0
        angle_psd = -math.atan2(dy, dx) * 180 / math.pi
        if shape_name == "linear":
            gradient_offset_x_psd_px = (end_x + start_x)/2  - w/2
            gradient_offset_y_psd_px = (end_y + start_y)/2  - h/2
            psd_gradient_type = 'Lnr '
        else:
            if (shape_name != "circular"):
                logging.warning("only linear and circular gradients export is supported, got %s. Forcing circular gradient type.", shape_name)
            scale_psd *= 2
            gradient_offset_x_psd_px = start_x - w/2
            gradient_offset_y_psd_px = start_y - h/2
            psd_gradient_type = 'Rdl '
        layer_offset1 = (layer.LayerOffsetX, layer.LayerOffsetY)
        layer_offset2 = (layer.LayerRenderOffscrOffsetX, layer.LayerRenderOffscrOffsetY)
        psd_gradient_offset_px = (gradient_offset_x_psd_px, gradient_offset_y_psd_px)
        logging.debug('gradient start--end point: %s -- %s, layer offsets: %s %s, shape: %s(%s), result psd offset in pixels: %s, width,height: %s', (start_x, start_y), (end_x, end_y), layer_offset1, layer_offset2, shape_name, gd.shape, psd_gradient_offset_px, (w, h))
        gradient_offset_x_psd = gradient_offset_x_psd_px * 100.0 / w
        gradient_offset_y_psd = gradient_offset_y_psd_px * 100.0 / h
        psd_gradient_offset = (gradient_offset_x_psd, gradient_offset_y_psd)
        assert len(psd_gradient_type) == 4 # PSD gradient binary uses 4 bytes for property values with space
        return (psd_gradient_offset, angle_psd, scale_psd, psd_gradient_type)

    def convert_color_stops_to_psd(color_stops, geometry_data):
        # Middle point is set always to 50% here. It affects gradient wolor-stops transition,
        # but it's complicated (but possible) to estimate middle point from ClipStudioPaint gradient curves.
        # So, at this moment curves are dropped, and middle point is always 50%.

        gd = geometry_data
        repeat_mode_name = { 0:"clip", 1:"repeat", 2:"mirror", 3:"transparent" }.get(gd.repeat_mode, "unknown_" + str(gd.repeat_mode))
        supported_modes = ("clip", "transparent")
        if repeat_mode_name not in supported_modes:
            logging.warning("gradient repeat mode '%s' is not supported, only %s are supported", repeat_mode_name, supported_modes)

        color_stops_psd = [(c.r, c.g, c.b, int(c.position / 100.0 * 4096), 50) for c in color_stops]

        transparency_stops_psd = [ [c.opacity, int(c.position/100*4096), 50] for c in color_stops ]
        trs = transparency_stops_psd
        if (repeat_mode_name == "transparent"):
            # add artifical transparency stops at start and end of list
            trs.insert(0, (0, 0, 0))
            trs.append((0, 4096, 100))
            i = 1
            # make transparency_stops different after modification
            while i < len(trs) and trs[i-1][1] >= trs[i][1]:
                trs[i][1] = trs[i-1][1] + (4096//100+1)
            i = len(trs) - 1
            while i > 0  and trs[i-1][1] >= trs[i][1]:
                trs[i-1][1] = trs[i][1] - (4096//100+1)
        trs.sort(key = lambda t: t[1]) # sort by positions
        return color_stops_psd, transparency_stops_psd


    def add_gradient_layer_info(gradient_info, layer_tags, layer):
        if not gradient_info:
            return

        logging.debug("exporting gradient")
        if gradient_info[0]:
            logging.info("exporting gradient as flat color")
            add_fill_color_for_layer(layer_tags, fill_color = gradient_info[1])
        else:
            _, (color_stops, geometry_data) = gradient_info
            (psd_gradient_offset, angle_psd, scale_psd, psd_gradient_type) = convert_clip_gradient_to_psd_gradient_math(geometry_data, layer)
            color_stops_psd, transparency_stops_psd = convert_color_stops_to_psd(color_stops, geometry_data)
            data = get_gradient_layer_binary_data_for_psd(psd_gradient_offset, angle_psd, scale_psd, psd_gradient_type, color_stops_psd, transparency_stops_psd)
            layer_tags.append((b'GdFl', data))

    def export_layer(f, layer_entry, make_invisible, txt=None, gradient_info=None):
        layer_type, layer = layer_entry
        logging.info("exporting '%s'", layer.LayerName if layer else '-')

        mask_data, layer_channels = write_bitmap_and_mask_channel_info_and_get_binary_data(f, layer_type, layer)

        lsct_folder_tag = None
        if layer_type == 'lt_folder_start':
            class FakeLayer:
                def __init__(self):
                    self.LayerName = '</Layer set>'
                    self.LayerComposite = 0
                    self.LayerLock = 0
                    self.LayerClip = 0
                    self.LayerOpacity= 0
                    self.LayerVisibility = 0
            layer = FakeLayer() # replace layer
            lsct_folder_tag = int(3).to_bytes(4, 'big')
        else:
            if layer_type == 'lt_folder_end':
                is_closed_folder = (layer.LayerFolder & 16)
                lsct_folder_tag = (2 if is_closed_folder else 1).to_bytes(4, 'big')

        blend_modes = {
            0:  'norm',  1: 'dark',  2: 'mul ',  3: 'idiv',  4: 'lbrn',  5: 'fsub',  6: 'dkCl',  7: 'lite',  8: 'scrn',  9: 'div ',
            10: 'div ', 11: 'lddg', 12: 'lddg', 13: 'lgCl', 14: 'over', 15: 'sLit', 16: 'hLit', 17: 'vLit', 18: 'lLit', 19: 'pLit',
            20: 'hMix', 21: 'diff', 22: 'smud', 23: 'hue ', 24: 'sat ', 25: 'colr', 26: 'lum ',
            30: 'pass', 36: 'fdiv'
        }

        blend_mode = blend_modes[layer.LayerComposite].encode('ascii')
        transparency_shapes_layer = (2**24) * (layer.LayerComposite not in (12, 10)) # clip's "Add(Glow)" and "glow dodge" blending modes
        if layer_type == 'lt_folder_start':
            transparency_shapes_layer = 0 # CSP exports folder start this way
        f.write(b'8BIM')
        f.write(blend_mode)

        if lsct_folder_tag != None:
            lsct_folder_tag += b'8BIM'
            lsct_folder_tag += blend_mode

        psd_opacity = int(layer.LayerOpacity*255/256 + 0.5) # This tries to replicate CSP->PSD export mapping [0..256] -> [0..255]. This function works for 3/4 of values, and gives +-1 diference on 1/4 of values.
        psd_opacity = max(0, min(psd_opacity, 255))
        write_int1(f, psd_opacity)
        write_int1(f, int(bool(layer.LayerClip)))

        locked_alpha = bool(layer.LayerLock & 16)
        layer_visible = bool(layer.LayerVisibility&1)

        if getattr(layer, 'OutputAttribute', False):
            layer_visible = False
            logging.info('disabling "Draft" layer visibility')

        layer_visible = layer_visible and (not make_invisible)

        layer_flags_byte = (not (layer_visible))*2 + locked_alpha*1
        write_int1(f, layer_flags_byte)
        # CSP has the bug: locked_alpha is reset to zero at export if layer edit is locked. This behaviour is not replicated.
        locked_edit = bool(layer.LayerLock & 1)
        lspf_layer_tag = 0x80000000 * locked_edit + 1 * locked_alpha
        write_int1(f, 0) # padding fill

        layer_extra_section = io.BytesIO()
        f_back = f
        f = layer_extra_section

        write_int(f, len(mask_data)) # mask section size
        f.write(mask_data)

        csp_better_match_output = b'\0\0\0\x28' + 10*b'\0\0\xff\xff'
        #write_int(f, 0) # layer blending ranges section size, not used
        f.write(csp_better_match_output) # some non-used 'blending ranges', just for better match of binary output of CSP export

        layer_name = layer.LayerName
        assert (isinstance(layer_name, str))
        write_layer_name(f, layer_name)

        layer_tags = [ ]

        add_fill_color_for_background_layer(layer_tags, layer)
        add_stroke_outline(layer_tags, layer)
        add_filter_layer_info(layer_tags, layer)
        add_gradient_layer_info(gradient_info, layer_tags, layer)
        add_layer_color_property(layer_tags, layer)

        if lsct_folder_tag != None:
            layer_tags.append(( b'lsct', lsct_folder_tag ))
        layer_tags.append(( b'lspf', lspf_layer_tag.to_bytes(4, 'big') ))
        # documentation requires 2 null bytes in the end of the string, but exported files from CSP dont use it,
        # and other psd files have mixed convention about these null bytes
        unicode_layer_name = len(layer_name).to_bytes(4, 'big') + layer_name.encode('UTF-16BE')
        layer_tags.append(( b'luni', unicode_layer_name ))
        layer_tags.append(( b'tsly', transparency_shapes_layer.to_bytes(4, 'big') ))
        if txt:
            layer_tags.append((b'TySh', txt))

        write_layer_tags(f, layer_tags)
        f = f_back
        extra_data_bytes = layer_extra_section.getvalue()

        write_int(f, len(extra_data_bytes))
        f.write(extra_data_bytes)
        return layer_channels

    def write_psd_header(f):
        channels = 3
        depth = 8
        color_mode = 3 # rgb

        header = b'8BPS' + struct.pack('>H6xHIIHH', psd_version, channels, int(sqlite_info.height), int(sqlite_info.width), depth, color_mode)
        assert len(header) == 26
        f.write(header)

    def get_dpi_resolution_dpi_data():
        f = io.BytesIO()
        logging.info('dpi: %s', sqlite_info.dpi)
        dpi_fixed = int(sqlite_info.dpi * 65536)
        dpi_int = dpi_fixed >> 16
        dpi_frac = dpi_fixed & 65536
        for _i in range(2):
            write_int2(f, dpi_int)
            write_int2(f, dpi_frac)
            write_int2(f, 1) # display unit unit 'dots per inch' for dpi resolution
            write_int2(f, 2) # display unit for width or height (2 - cm)
        return f.getvalue()

    def write_image_resource_section_entry(f, entry_id, data):
        f.write(b'8BIM')
        write_int2(f, entry_id)
        write_int2(f, 0) # empty name - two 0 bytes
        write_int(f, len(data))
        f.write(data)

    def write_image_resources_section(f):
        section_data_stream = io.BytesIO()
        write_image_resource_section_entry(section_data_stream, 1005, get_dpi_resolution_dpi_data())
        section_data = section_data_stream.getvalue()
        write_int(f, len(section_data))
        f.write(section_data)

    def get_text_aligment(text, clip_text_params):
        # assume whole text has one common paragraph alignment, because it's impossible to export text from 
        # Clip to PSD with different alignment of different parts of the text: Clip assumes that 
        # left/right/center alignment is aligned to the left/rigt/center of the box, PSD assumes single vertical 
        # line is common for all alignment types. So this line needs to be shifted to keep text at the same 
        # position as Clip has, and it can't be shifted it to different directions at the same time for different 
        # parts of the text. 
        # It's theoretically possible to write each aligned paragraph as different text section, but it would require
        # perfect pixel matching of rendered font height in Clip and Photoshop. Also it's possible to set paragraph shift, 
        # but this unusual option possibly will confuse user.
        #
        # So, select aligment based on maximum number of letters for each aligment.
        align_total_length = { }
        for a in clip_text_params['param_align']:
            l = (min(len(text), a.start + a.length) - max(a.start, 0))
            align_total_length[a.align] = align_total_length.get(a.align, 0) + l

        if align_total_length:
            align = max(sorted(align_total_length.keys()), key = lambda a: align_total_length[a])
        else:
            align = 0
        return align

    def calc_psd_text_transform_matrix(layer_offset, text_align, first_line_height, clip_text_params):
        bbox = clip_text_params['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        wh = [w, h]
        dx, dy = 0, 0

        if text_align == 2: # align center 
            dx += w // 2 
        elif text_align == 1: # align right
            dx += w

        dy += first_line_height

        q = clip_text_params['quad_verts']
        a = [q[2+i] - q[0+i] for i in (0,1)] # a, b - vectors along parallelogram sides
        b = [q[6+i] - q[0+i] for i in (0,1)]

        # multiply two 2x3 matrix, assume third row is [0,0,1]
        def multiply(a, b):
            def mul_col(j, b_j2):
                return [sum(a[k][i] * b[j][k] for k in range(2)) + a[2][i]*b_j2 for i in range(2)]

            return [mul_col(0, 0.0), mul_col(1, 0.0), mul_col(2, 1.0)]

        def multiply_matrix_list(*xs):
            y = xs[0]
            for i in range(1, len(xs)):
                y = multiply(y, xs[i])
            return y

        def move(dx, dy): return [[1, 0], [0, 1], [dx, dy]]
        def scale(sx, sy): return [[sx, 0], [0, sy], [0, 0]]

        # interpret quad_verts as parallelogram, buid matrix to map bbox to quad_verts with the common center. 
        # ClipStudio upscales font size for vertical  stretch/squeze scale, but keeps it the same for horizontal <-> scale. 
        # So leave vertical scale handling to font size, but horizontal scale is handled by matrix. 
        hor_scale = clip_text_params['aspect_ratio']/100.0 
        logging.debug('hor_scale %s', hor_scale)

        offset = [(wh[i] - a[i] - b[i])/2 + bbox[i] + layer_offset[i] - q[i] for i in (0,1)] 
        matrix = multiply_matrix_list([a, b, offset], scale(1/w, 1/h), move(dx, dy), scale(hor_scale, 1))
        return matrix
        

    def calc_text_bounding_box(layer_offset, clip_text_params, matrix):
        text_align2 = 0
        first_line_height2 = 0.0
        matrix2 = calc_psd_text_transform_matrix(layer_offset, text_align2, first_line_height2, clip_text_params)
        origin = matrix[2]

        bbox = clip_text_params['bbox']
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        result_quad = [[matrix2[0][i]*x + matrix2[1][i]*y + matrix2[2][i] - origin[i] for i in (0,1)] for x in (0,w) for y in (0,h)]
        xs, ys = zip(*result_quad)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, max_x, min_y, max_y

    def convert_to_mac_eol(text, style_list):
        result, result_style = [], []
        i = 0
        while i < len(text):
            c = text[i]
            result_style.append(style_list[i])
            if c == '\r' and i+1 < len(text) and text[i+1] == '\n':
                i += 1
            if c == '\n':
                c = '\r'
            result.append(c)
            i += 1

        return ''.join(result), result_style

    def fill_text_letters_styles(text_str, clip_text_params):
        default_font_display_name = clip_text_params['font']

        default_font_size = clip_text_params['font_size'] / 100.0 * (sqlite_info.dpi / 72.0)
        default_style = {
            'font_size': default_font_size,
            'font_display_name': default_font_display_name,
            'color': clip_text_params['color'],
            'auto_kerning': 'true',
            'style_flags': 0,
            'underline': False,
            'strike': False,
        }

        index2style = [dict(default_style) for _ in range(len(text_str))]
        index2nondefault = [False for _ in range(len(index2style))]

        def get_checked_start_end(obj_range):
            start = obj_range.start
            if start < 0:
                logging.warning("invalid sequence start %d for text %s", start, repr(text_str))
                start = 0
            end = obj_range.start + obj_range.length
            if not (end <= len(index2style)):
                logging.warning("invalid sequence end %d (%d+%d) for text %s", end, obj_range.start, obj_range.length, repr(text_str))
                end = len(index2style)
            return start, end

        for run in clip_text_params['runs']:
            font_display_name = default_font_display_name
            if run.font:
                font_display_name = run.font

            color = run.color if (run.field_defaults_flags & 1) else clip_text_params['color']
            for c in color:
                if not (0.0 <= c <= 1.0):
                    logging.warning('invalid color out of range, %s', color)
            color = tuple(max(min(1.0, c), 0.0) for c in color)
            font_size = default_font_size
            if run.field_defaults_flags & 2:
                if run.font_scale == 0.0:
                    logging.warning("zero font scale, ingored")
                else:
                    font_size *= run.font_scale/100.0
            elif run.font_scale != 0.0:
                logging.warning("non-zero font scale without scale flag")

            start, end = get_checked_start_end(run)

            for i in range(start, end):
                t = index2style[i]
                if index2nondefault[i]:
                    logging.warning('conflicting styles at %d for text %s', i, repr(text_str))
                index2nondefault[i] = True
                t['font_size'] = font_size
                t['font_display_name'] = font_display_name
                t['color'] = color
                t['style_flags'] = run.style_flags & 3

        del index2nondefault
        index2style[0]['auto_kerning'] = 'false' # mimic psd file exported from photoshop

        for clip_param_name, style_param_name in [('param_underline', 'underline'), ('param_strike', 'strike')]:
            for r in clip_text_params.get(clip_param_name, []):
                start, end = get_checked_start_end(r)
                for i in range(start, end):
                    index2style[i][style_param_name] = True


        for i in range(len(text_str)):
            if not index2style[i]:
                logging.warning('empty style at index %d for text %s', i, repr(text_str))

        return index2style

    def encode_psd_text_engine_unicode(text):
        text_1byte_coded = text.encode('UTF-16BE').decode('latin-1') 
        text_escaped = text_1byte_coded.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        return '(\u00FE\u00FF' + text_escaped + ')'

    class FontIndexGenerator:
        def __init__(self, clip_text_params):
            self.font_display_name_to_name = {}
            self.font_name_to_index = {}

            for font_display_name, font_name in clip_text_params['fonts'].font_list:
                self.font_display_name_to_name[font_display_name] = font_name

        def get_font_index_and_leftover_style_non_cached(self, font_display_name, style):
            font_name = self.font_display_name_to_name.get(font_display_name)
            if font_name == None:
                logging.warning('font %s not in font list %s', repr(font_display_name), repr(sorted(self.font_display_name_to_name)))
                font_name = 'Tahoma'

            assert 0 <= style <= 3
            if style == 0:
                return font_name, 0

            font_style_table = table_regular_font_to_styled.get(font_name, [None, None, None])
            font_name2 = font_style_table[style-1]
            if font_name2:
                return font_name2, 0

            if style != 3 or (font_style_table[0] == font_style_table[1] == None):
                # no other options to try
                return font_name, style

            if font_style_table[0]:
                return font_style_table[0], 2
            if font_style_table[1]:
                return font_style_table[1], 1

            return font_name, style

        def add_font_and_get_index(self, font_name):
            if font_name not in self.font_name_to_index:
                new_index = len(self.font_name_to_index)
                self.font_name_to_index[font_name] = new_index
            return self.font_name_to_index[font_name]

        def get_font_index_and_leftover_style(self, font_display_name, style):
            font_name, style_leftover = self.get_font_index_and_leftover_style_non_cached(font_display_name, style)
            return self.add_font_and_get_index(font_name), style_leftover


    def generate_style_and_fonts_text_engine_script_parts(index2style, clip_text_params):
        font_index_generator = FontIndexGenerator(clip_text_params)

        style_array_psd_text = ''
        style_array_psd_lengths = []
        for style, indices in itertools.groupby(range(len(index2style)), key = index2style.__getitem__):
            font_size_str = ('%.2f' % (style['font_size']))
            if font_size_str.endswith('.00'):
                font_size_str = font_size_str[:-1] # "48.00 -> 48.0
            t = psd_text_engine_style_template.replace('{style_font_size}', font_size_str)
            def format_color(c):
                # mimic formatting in psd
                if c == 0: return '0.0'
                if c == 1: return '1.0'
                c = min(1.0, c)
                cint = int(c * 255 + 0.5) # restore original clip exact integer values
                ctxt = '%.5f' % (cint / 255.0) # convert them to psd
                if ctxt[0:2] == '0.':
                    ctxt = ctxt[1:]
                return ctxt

            font_index, new_font_style_flags = font_index_generator.get_font_index_and_leftover_style(style['font_display_name'], style['style_flags'])
            bold = new_font_style_flags & 1
            italic = new_font_style_flags & 2

            t = t.replace('{style_color}', ' '.join(map(format_color, style['color']))) 
            t = t.replace('{style_font_index}', str(font_index))
            t = t.replace('{style_auto_kerning}', style['auto_kerning'])
            t = t.replace('{style_bold}', psd_text_bold if bold else '')
            t = t.replace('{style_italic}', psd_text_italic if italic else '')
            t = t.replace('{style_underline}', psd_text_underline if style['underline'] else '')
            t = t.replace('{style_strike}', psd_text_strike if style['strike'] else '')
            style_array_psd_text += t
            style_array_psd_lengths.append(len(list(indices)))

        # add these fonts for closer matching to real PSD example
        font_index_generator.add_font_and_get_index('AdobeInvisFont') 
        font_index_generator.add_font_and_get_index('Tahoma')
        font_set_psd_text = ''
        for font_name, _index in sorted(font_index_generator.font_name_to_index.items(), key = lambda ab: (ab[1], ab[0])):
            t = psd_text_engine_font_template.replace('{font_type}', str(int(font_name != 'AdobeInvisFont')))
            t = t.replace('{font_name}', encode_psd_text_engine_unicode(font_name))
            font_set_psd_text += t

        return style_array_psd_text, style_array_psd_lengths, font_set_psd_text 

    def make_text_engine_script(text_str, text_align, clip_text_params):
        logging.debug("text before export: %s", repr(text_str))
        index2style_for_original_text = fill_text_letters_styles(text_str, clip_text_params)
        assert len(text_str) == len(index2style_for_original_text)
        exported_text, index2style = convert_to_mac_eol(text_str, index2style_for_original_text)
        assert len(exported_text) == len(index2style)

        text_paragraphs = [(x + '\r') for x in exported_text.split('\r')] # also implicitly adds extra '\r' 

        paragraph_list_text_engine = ''
        paragraph_lengths = []
        for p in text_paragraphs:
            paragraph_list_text_engine += paragraph_template.replace('{style_paragraph_align}', str(text_align))
            paragraph_lengths.append(str(len(p)))

        exported_text += '\r'
        index2style.append(index2style[-1]) # extra '\r' was added, transfer style from previous letter
        assert len(exported_text) == len(index2style)
        logging.debug("result exported text: %s", repr(exported_text))
        assert exported_text == ''.join(text_paragraphs)

        first_line_end = exported_text.find('\r') + 1
        first_line_height = max(index2style[i]['font_size'] for i in range(first_line_end))

        style_array_psd_text, style_array_psd_lengths, font_set_psd_text = generate_style_and_fonts_text_engine_script_parts(index2style, clip_text_params)

        txt = psd_text_engine_script
        txt = txt.replace('{paragraph_list}', paragraph_list_text_engine)
        txt = txt.replace('{paragraph_list_lenghts}', '[ %s ]' % ' '.join(paragraph_lengths))
        txt = re.sub("'(.*)'", lambda m: encode_psd_text_engine_unicode(m.group(1)), txt)
        txt = txt.replace('{font_set}', font_set_psd_text) 
        txt = txt.replace('{style_run_array}', style_array_psd_text) 
        txt = txt.replace('{style_run_array_lengths}', ' '.join(map(str, style_array_psd_lengths))) 
        # this text can contain any characters, including my replace markers or escape symbols, so it must be inserted with care as last operation. Same about font names, but they are more regular.
        txt = txt.replace('{exported_text}', encode_psd_text_engine_unicode(exported_text)) 
        text_engine_script_result_binary = txt.encode('latin-1')

        return text_engine_script_result_binary, first_line_height

    def write_matrix_and_get_text_bounds(f, layer_offset, text_str, clip_text_params):
        text_align = get_text_aligment(text_str, clip_text_params)
        text_engine_script_result_binary, first_line_height = make_text_engine_script(text_str, text_align, clip_text_params)
        matrix = calc_psd_text_transform_matrix(layer_offset, text_align, first_line_height, clip_text_params)
        min_x, max_x, min_y, max_y = calc_text_bounding_box(layer_offset, clip_text_params, matrix)

        logging.debug('matrix for text transform: %s', matrix)
        for col in matrix:
            for x in col:
                f.write(struct.pack('>d', x))

        return min_x, max_x, min_y, max_y, text_engine_script_result_binary

    class PsdObjDescriptorWriter:
        def __init__(self, f):
            self.f = f

        def write_key(self, x, write4as0 = True):
            write_int(self.f, (0 if (len(x) == 4 and write4as0) else len(x)))
            self.f.write(x.encode('ascii'))
        def write_obj_header_named(self, class_id, field_count, class_name, write4as0 = True):
            self.write_unicode_str(class_name)
            self.write_key(class_id, write4as0)
            write_int(self.f, field_count) # obj field count
        def write_obj_header(self, class_id, field_count, write4as0 = True):
            self.write_obj_header_named(class_id, field_count, class_name = '', write4as0 = write4as0)
        def write_list_item(self, typename):
            self.f.write(typename.encode('ascii'))
        def write_field(self, field, field_type, write4as0 = True):
            self.write_key(field, write4as0)
            assert len(field_type) == 4
            self.f.write(field_type.encode('ascii'))
        def write_enum2(self, field_name, enum_type, value):
            self.write_field(field_name, 'enum')
            self.write_key(enum_type)
            self.write_key(value)
        def write_enum(self, field_name, value):
            self.write_enum2(field_name, field_name, value)
        def write_unicode_str(self, s):
            write_int(self.f, len(s) + 1)
            self.f.write(s.encode('UTF-16BE'))
            write_int2(self.f, 0) # null char
        def write_doub(self, field_name, value):
            self.write_field(field_name, 'doub')
            self.f.write(struct.pack('>d', value))
        def write_untf(self, field_name, unit_name, value):
            self.write_field(field_name, 'UntF')
            self.f.write(unit_name.encode('ascii'))
            self.f.write(struct.pack('>d', value))
        def write_long(self, field_name, value, write4as0 = True):
            self.write_field(field_name, 'long', write4as0)
            write_int(self.f, value)
        def write_bool(self, field_name, value, write4as0 = True):
            self.write_field(field_name, 'bool', write4as0)
            write_int1(self.f, int(bool(value)))

    def make_psd_text_layer_property(layer_offset, text_str, clip_text_params):
        f = io.BytesIO()
        write_int2(f, 1) # version

        min_x, max_x, min_y, max_y, text_engine_script_result_binary = write_matrix_and_get_text_bounds(f, layer_offset, text_str, clip_text_params)

        write_int2(f, 50) # TySh text engine version
        write_int(f, 16) # descriptor version

        obj = PsdObjDescriptorWriter(f)

        obj.write_obj_header('TxLr', 8)
        obj.write_field('Txt ', 'TEXT')
        obj.write_unicode_str(text_str)
        obj.write_enum('textGridding', 'None')
        obj.write_enum('Ornt', 'Hrzn') # Orientation of text #TODO: try export vertical text
        obj.write_enum2('AntA', 'Annt', 'antiAliasSharp')

        def write_bounds(class_id, left, top, right, bottom):
            field_name = class_id
            obj.write_field(field_name, 'Objc')
            obj.write_obj_header(class_id, 4)
            obj.write_untf('Left', '#Pnt', left)
            obj.write_untf('Top ', '#Pnt', top)
            obj.write_untf('Rght', '#Pnt', right)
            obj.write_untf('Btom', '#Pnt', bottom)

        write_bounds('bounds', min_x, min_y, max_x, max_y)
        write_bounds('boundingBox', min_x, min_y, max_x, max_y)
        obj.write_long('TextIndex', 0)
        obj.write_field('EngineData', 'tdta')

        write_int(f, len(text_engine_script_result_binary))
        f.write(text_engine_script_result_binary)

        write_int2(f, 1) # warp version
        write_int(f, 16) # descriptor version
        obj.write_obj_header('warp', 5, write4as0 = False)
        obj.write_enum('warpStyle', 'warpNone')
        obj.write_doub('warpValue', 0)
        obj.write_doub('warpPerspective', 0)
        obj.write_doub('warpPerspectiveOther', 0)
        obj.write_enum2('warpRotate', 'Ornt', 'Hrzn')
        for _i in range(4):
            write_int(f, 0) # left=0 top=0 right=0 bottom=0, doc doesn't say what is this rectangle, values are taken from psd example

        data = f.getvalue()
        data += bytes(-len(data) % 4) # padding 4 bytes
        return data

    def export_layer_text(layer_entry):
        _layer_type, layer = layer_entry

        def split_array(data):
            result = []
            i = 0
            while i < len(data):
                s = data[i:i+4]
                if len(s) != 4:
                    raise ValueError("can't read int from test layer settings")
                l = int.from_bytes(s, 'little')
                i += 4
                if i + l > len(data):
                    raise ValueError(f"got too large size from test layer settings {i}+{l} > {len(data)}")
                result.append(data[i:i+l])
                i += l
            assert i == len(data)
            return result

        def get_array(field_name):
            first = getattr(layer, field_name, None)
            array_data = getattr(layer, field_name + "Array", None)
            result = []
            if first != None:
                result.append(first)
            if array_data != None:
                result.extend(split_array(array_data))
            return result

        text_strings = [t.decode('UTF-8') for t in get_array('TextLayerString')]
        text_attributes = get_array('TextLayerAttributes')

        if len(text_strings) != len(text_attributes):
            logging.warning("length mismatch of text_strings(%s) and text_attributes(%s) arrays in layer", len(text_strings), len(text_attributes))

        for text_str in text_strings:
            logging.debug('text_str 1: %s', repr(text_str)) # duplicated debug output, because zip() can truncate array

        text_layer_tags = []
        for text_str, t in zip(text_strings, text_attributes):
            layer_offset = layer.LayerOffsetX, layer.LayerOffsetY
            logging.debug('text_str 2: %s', repr(text_str))
            clip_text_params = parse_layer_text_attribute(t)
            if 'quad_verts' not in clip_text_params:
                logging.warning("text does not have bounding quadrangle, skipping")
                continue
            text_layer_tags.append( make_psd_text_layer_property(layer_offset, text_str, clip_text_params) )
        return text_layer_tags

    def write_layers_data_section(f):
        layers_full_section_start = f.tell()
        write_int_psb(f, 0) # placeholder for size
        layers_info_subsection_start = f.tell()
        write_int_psb(f, 0) # placeholder for size
        layer_count_position = f.tell()
        write_int2(f, 0) # placeholder for layer_count

        channels_data = []
        for layer_entry in layer_ordered:
            l = layer_entry[1]
            if l != None:
                logging.debug('layer_offset: %s', [l.LayerOffsetX, l.LayerOffsetY])
                logging.debug('layer_render_offset: %s', [l.LayerRenderOffscrOffsetX, l.LayerRenderOffscrOffsetY])

            text_info = []
            if not (cmd_args.text_layer_raster != 'enable' and cmd_args.text_layer_vector == 'disable'):
                # don't parse text data, if command line arguments don't require this
                text_info = export_layer_text(layer_entry)

            gradient_info = None
            if not (cmd_args.gradient_layer_raster != 'enable' and cmd_args.gradient_layer_vector == 'disable'):
                gradient_bytes_data = getattr(l, "GradationFillInfo", None)
                if gradient_bytes_data:
                    gradient_info = parse_gradation_fill_data_of_gradient_layers(gradient_bytes_data)

            # don't manage raster/vector export of flat color layers with command line options targeted at gradients.
            # They never has pixel data in .clip files, so attempt to get perfect original gradient layers pixel
            # data by request of rasterized layer could work for non-flat gradient layers, but definitely will corrupt flat
            # layers export at the same time.
            is_gradient_layer = bool(gradient_info) and (False == gradient_info[0])
            is_flat_color_layer = bool(gradient_info) and (True == gradient_info[0])

            disabled_raster_because_text = bool(text_info) and cmd_args.text_layer_raster == "disable"
            disabled_raster_because_gradient = (is_gradient_layer and cmd_args.gradient_layer_raster == "disable")
            invisible_because_text = bool(text_info) and cmd_args.text_layer_raster == "invisible"
            invisible_because_gradient = is_gradient_layer and cmd_args.gradient_layer_raster == "invisible"

            if not (disabled_raster_because_text or disabled_raster_because_gradient or is_flat_color_layer):
                make_invisible = invisible_because_text or invisible_because_gradient
                channels_data.append(export_layer(f, layer_entry, make_invisible, None))

            if cmd_args.text_layer_vector != 'disable':
                _, layer = layer_entry
                if text_info:
                    logging.info("exporting '%s' as text layer", layer.LayerName if layer else '-')
                for txt in text_info:
                    make_invisible = cmd_args.text_layer_vector == 'invisible'
                    channels_data.append(export_layer(f, ('lt_text', layer), make_invisible, txt = txt))

            if is_gradient_layer and cmd_args.gradient_layer_vector != 'disable':
                _, layer = layer_entry
                make_invisible = cmd_args.gradient_layer_vector == 'invisible'
                channels_data.append(export_layer(f, ('lt_gradient', layer), make_invisible, gradient_info = gradient_info))

            if is_flat_color_layer:
                channels_data.append(export_layer(f, ('lt_gradient', layer), make_invisible = False, gradient_info = gradient_info))


        for layer_channels in channels_data:
            for file_data in layer_channels:
                f.write(file_data)

        layers_info_subsection_end = f.tell()
        write_int(f,  0) # global layer mask (skipped)
        # tagged layers info sections - skipped
        #f.write(global_text_tag[4:])

        end = f.tell()

        f.seek(layer_count_position)
        layer_count = len(channels_data)
        write_int2(f, layer_count)

        f.seek(layers_full_section_start)
        size_of_size = psd_version * 4
        write_int_psb(f, end - layers_full_section_start - size_of_size)

        f.seek(layers_info_subsection_start)
        write_int_psb(f, layers_info_subsection_end - layers_info_subsection_start - size_of_size)
        f.seek(end) # padding?..



    def export_psd():
        with open(output_psd, 'wb') as f:
            write_psd_header(f)
            write_int(f, 0) #Color Mode Data section (empty)
            #write_int(f, 0) #Image Resources section (empty)
            write_image_resources_section(f)
            write_layers_data_section(f)
            export_canvas_preview(f)

    export_psd()


def extract_csp(filename):
    with open(filename, 'rb') as f:
        data = f.read()

    file_chunks_list = iterate_file_chunks(data, filename)
    for chunk_name, chunk_data_memory_view, _chunk_offset in file_chunks_list:
        if chunk_name == b'SQLi':
            logging.info('writing .clip sqlite database at "%s"', cmd_args.sqlite_file)
            with open(cmd_args.sqlite_file, 'wb') as f:
                f.write(chunk_data_memory_view)

    sqlite_info = get_sql_data_layer_chunks()

    id2layer = { l.MainId:l for l in sqlite_info.layer_sqlite_info }
    layer_ordered = [ ]
    def print_layer_folders(folder_id, depth):
        folder = id2layer[folder_id]
        current_id = folder.LayerFirstChildIndex
        while current_id:
            l = id2layer[current_id]
            is_subfolder = l.LayerFolder != 0
            logging.info('%s %s', (depth*4)*' ' + ('*' if is_subfolder else ' '), l.LayerName)
            if is_subfolder:
                layer_ordered.append(('lt_folder_start', None))
                print_layer_folders(current_id, depth + 1)
                layer_ordered.append(('lt_folder_end', l))
            else:
                layer_ordered.append(('lt_bitmap', l))
            current_id = l.LayerNextIndex

    if cmd_args.output_psd or cmd_args.output_dir:
        logging.info('Layers names in tree:')
        print_layer_folders(sqlite_info.root_folder, 0)

    chunk_to_layers = {}
    for ofs in sqlite_info.offscreen_chunks_sqlite_info:
        chunk_to_layers.setdefault(ofs.BlockData, set()).add(ofs.LayerId)
    for v in sqlite_info.vector_info:
        chunk_to_layers.setdefault(v.VectorData, set()).add(v.LayerId)
    for k, v in chunk_to_layers.items():
        chunk_to_layers[k] = sorted(v)
    layer_names = {}
    for layer in sqlite_info.layer_sqlite_info:
        layer_names[layer.MainId] = layer.LayerName

    chunks = extract_csp_chunks_data(file_chunks_list, cmd_args.output_dir, chunk_to_layers, layer_names)

    if cmd_args.output_preview_image:
        save_preview_image(cmd_args.output_preview_image, sqlite_info)

    if cmd_args.output_dir:
        save_layers_as_png(chunks, cmd_args.output_dir, sqlite_info)
        #TODO: json with layer structure?..

    if cmd_args.output_psd:
        save_psd(cmd_args.output_psd, chunks, sqlite_info, layer_ordered)

    if not cmd_args.keep_sqlite:
        os.unlink(cmd_args.sqlite_file)

# Initialize global variable for the command line result object
cmd_args = None

def init_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(level=numeric_level)

def parse_command_line():
    global cmd_args # pylint: disable=global-statement

    parser = argparse.ArgumentParser(
        description='Convert Clip Studio Paint files to PSD or PSB format.\nBasic usage: python clip_to_psd.py input.clip -o output.psd',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('input_file', help='Input CLIP file to be converted.', type=str) # Mandatory input file name
    parser.add_argument('-o', '--output-psd', help='Output PSD file name.', type=str)
    parser.add_argument('--psd-version', help='Version of output PSD: 1 for PSD, 2 for PSB.', type=int, choices=[1, 2], default=1)
    parser.add_argument('--output-dir', help='Output directory to save layers as PNG files.', type=str)

    parser.add_argument('--log-level', help='Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).', type=str, default='INFO')
    parser.add_argument('--sqlite-file', help='Path for the output SQLite file, imply --keep-sqlite', type=str)
    parser.add_argument('--output-preview-image', help='Path for image preview export, example: "./preview.png". Note: preview is downscaled version of canvas.', type=str)
    parser.add_argument('--keep-sqlite', help='Keep the temporary SQLite file, path is derived from output psd or output directory if not specified with --sqlite-file', action='store_true')
    parser.add_argument('--text-layer-raster', help='Export text layers as raster: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='enable')
    parser.add_argument('--text-layer-vector', help='Export text layers as vector: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='invisible')
    parser.add_argument('--gradient-layer-raster', help='Export gradient layers as raster: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='enable')
    parser.add_argument('--gradient-layer-vector', help='Export gradient layers as vector: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='invisible')
    parser.add_argument('--ignore-zlib-errors', help='ignore decompression error for damaged data', action='store_true')
    parser.add_argument("--blank-psd-preview", help="--don't generate psd thumbnail preview (this allows to avoid Image module import, and works faster/gives smaller output file)", action='store_true')
    parser.add_argument('--psd-empty-bitmap-data', help='export bitmaps as empty to psd, usefull for faster export when pixel data is not needed', action='store_true')

    cmd_args = parser.parse_args()

    init_logging(cmd_args.log_level)

    logging.debug('command line: %s', sys.argv)

    if cmd_args.sqlite_file:
        cmd_args.keep_sqlite = True

    outputs = [(cmd_args.sqlite_file if cmd_args.keep_sqlite else None), cmd_args.output_dir, cmd_args.output_psd, cmd_args.output_preview_image]

    if not any(outputs):
        cmd_args.output_psd = os.path.splitext(cmd_args.input_file)[0] + '.psd'
        outputs.append(cmd_args.output_psd)
        if os.path.isfile(cmd_args.output_psd):
            logging.error("output file '%s' already exists, would not overwrite if not explicitly requested by -o option.", cmd_args.output_psd)
            sys.exit(1)
    outputs = [f for f in outputs if f]


    if cmd_args.output_psd:
        ext = os.path.splitext(cmd_args.output_psd.lower())[1]
        if ext.lower() not in ('.psd', '.psb'):
            logging.warning("unexpected extension '%s' for psd file, expected .psd or .psb", ext)
        if ext.lower() == '.psb' and cmd_args.psd_version != 2:
            logging.warning("unexpected extension version=%s for psb file, psb (Photosho Big File) are usually have version 2", cmd_args.psd_version)


    # If sqlite-file is not specified, derive it from the output PSD file or output directory
    if not cmd_args.sqlite_file:
        if cmd_args.output_psd:
            cmd_args.sqlite_file = os.path.splitext(cmd_args.output_psd)[0] + '.sqlite'
        elif cmd_args.output_dir:
            cmd_args.sqlite_file = os.path.join(cmd_args.output_dir, 'output.sqlite')
        elif cmd_args.output_preview_image:
            cmd_args.sqlite_file = os.path.splitext(cmd_args.output_preview_image)[0] + '.sqlite'
        else:
            parser.error('Output SQLite file path cannot be derived. Please specify --sqlite-file or an output option.')

    if cmd_args.output_dir:
        if not os.path.isdir(cmd_args.output_dir):
            os.mkdir(cmd_args.output_dir)

    return outputs



def main():
    outputs = parse_command_line()
    extract_csp(cmd_args.input_file)

    logging.info("export done to %s", ', '.join(f"'{x}'" for x in outputs))

if __name__ == "__main__":
    main()
