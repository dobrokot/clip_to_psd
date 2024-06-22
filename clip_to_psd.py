

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
# import Image from PIL:
# also imports Image from PIL if command line requires this. module Image is not loaded, if --blank-psd-preview is used, and no output dir for PNGs. So it's possible to export PSD with only built-in python modules.
import itertools
import argparse


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
                layer_name_sanitized = re.sub(r'''[\0-\x1f'"/\\:*?<>| ]''', '_', layer_name) 
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

def parse_offscreen_attributes_sql_value(offscreen_attribute):
    b_io = io.BytesIO(offscreen_attribute)
    def get_next_int(): return int.from_bytes(b_io.read(4), 'big')
    def check_read_str(s):
        str_size = get_next_int()
        assert str_size == len(s), (len(s), str_size, repr(offscreen_attribute))
        str2bytes = b_io.read(2*str_size)
        assert s.encode('UTF-16-BE') == str2bytes, (s, repr(str2bytes), repr(offscreen_attribute))

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

    def get_layer_chunk_data(mipmap_id):
        if mipmap_id:
            external_block_row = offscreen_dict[mipmapinfo_dict[mipmap_dict[mipmap_id].BaseMipmapInfo].Offscreen]
            external_id = external_block_row.BlockData
            if external_id not in chunks:
                return None
            return (chunks[external_id].bitmap_blocks, external_block_row.Attribute)
        return None

    def xxx_check(mipmap_id):
        if mipmap_id:
            external_block_row = offscreen_dict[mipmapinfo_dict[mipmap_dict[mipmap_id].BaseMipmapInfo].Offscreen]
            external_id = external_block_row.BlockData
            print(external_id in chunks, external_id)
    for l in sqlite_info.layer_sqlite_info:
        print()
        print('aaaa', l.LayerName)
        xxx_check(l.LayerRenderMipmap)
        xxx_check(l.LayerLayerMaskMipmap)

    for l in sqlite_info.layer_sqlite_info:
        layer_bitmaps[l.MainId] = LayerBitmaps(
            get_layer_chunk_data(l.LayerRenderMipmap),
            get_layer_chunk_data(l.LayerLayerMaskMipmap))

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

psd_text_engine_data = (
 b'\n\n<<\n\t/EngineDict\n\t<<\n\t\t/Editor\n\t\t<<\n\t\t\t/Text (\xfe'
 b'\xff\x00t\x00e\x00s\x00t\x00 \x00s\x00t\x00r\x00i\x00n\x00g\x00\r\x00o\x00'
 b't\x00h\x00e\x00r\x00 \x00l\x00i\x00n\x00e\x00\r\x00t\x00h\x00i\x00r\x00'
 b'd\x00 \x00l\x00i\x00n\x00e\x00\r)\n\t\t>>\n\t\t/ParagraphRun\n\t\t<<\n\t\t\t'
 b'/DefaultRunData\n\t\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/D'
 b'efaultStyleSheet 0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t>>\n'
 b'\t\t\t\t>>\n\t\t\t\t/Adjustments\n\t\t\t\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 '
 b']\n\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t/RunArray [\n\t'
 b'\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/DefaultStyleSheet '
 b'0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justification 2\n\t\t'
 b'\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t\t\t/StartIndent 0.0\n\t\t\t\t\t\t/E'
 b'ndIndent 0.0\n\t\t\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t/SpaceAfter 0.'
 b'0\n\t\t\t\t\t\t/AutoHyphenate true\n\t\t\t\t\t\t/HyphenatedWordSize 6\n'
 b'\t\t\t\t\t\t/PreHyphen 2\n\t\t\t\t\t\t/PostHyphen 2\n\t\t\t\t\t\t/Consecutiv'
 b'eHyphens 8\n\t\t\t\t\t\t/Zone 36.0\n\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33'
 b' ]\n\t\t\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n\t\t\t\t\t\t/GlyphSpacing '
 b'[ 1.0 1.0 1.0 ]\n\t\t\t\t\t\t/AutoLeading 1.2\n\t\t\t\t\t\t/LeadingType 0\n'
 b'\t\t\t\t\t\t/Hanging false\n\t\t\t\t\t\t/Burasagari false\n\t\t\t\t\t\t/Kins'
 b'okuOrder 0\n\t\t\t\t\t\t/EveryLineComposer false\n\t\t\t\t\t>>\n\t\t\t\t>>'
 b'\n\t\t\t\t/Adjustments\n\t\t\t\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n'
 b'\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t<<\n\t\t\t\t/Paragra'
 b'phSheet\n\t\t\t\t<<\n\t\t\t\t\t/DefaultStyleSheet 0\n\t\t\t\t\t/Propertie'
 b's\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justification 2\n\t\t\t\t\t\t/FirstLineIndent '
 b'0.0\n\t\t\t\t\t\t/StartIndent 0.0\n\t\t\t\t\t\t/EndIndent 0.0\n\t\t\t\t'
 b'\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t/SpaceAfter 0.0\n\t\t\t\t\t\t/AutoHyphenat'
 b'e true\n\t\t\t\t\t\t/HyphenatedWordSize 6\n\t\t\t\t\t\t/PreHyphen 2\n\t\t'
 b'\t\t\t\t/PostHyphen 2\n\t\t\t\t\t\t/ConsecutiveHyphens 8\n\t\t\t\t\t\t/Zon'
 b'e 36.0\n\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t\t\t/LetterSpaci'
 b'ng [ 0.0 0.0 0.0 ]\n\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t\t\t'
 b'\t/AutoLeading 1.2\n\t\t\t\t\t\t/LeadingType 0\n\t\t\t\t\t\t/Hanging false\n'
 b'\t\t\t\t\t\t/Burasagari false\n\t\t\t\t\t\t/KinsokuOrder 0\n\t\t\t\t\t\t/Eve'
 b'ryLineComposer false\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t/Adjustments\n\t\t\t'
 b'\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t'
 b'>>\n\t\t\t>>\n\t\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/Defa'
 b'ultStyleSheet 0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justific'
 b'ation 2\n\t\t\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t\t\t/StartIndent 0.'
 b'0\n\t\t\t\t\t\t/EndIndent 0.0\n\t\t\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t'
 b'/SpaceAfter 0.0\n\t\t\t\t\t\t/AutoHyphenate true\n\t\t\t\t\t\t/HyphenatedW'
 b'ordSize 6\n\t\t\t\t\t\t/PreHyphen 2\n\t\t\t\t\t\t/PostHyphen 2\n\t\t\t'
 b'\t\t\t/ConsecutiveHyphens 8\n\t\t\t\t\t\t/Zone 36.0\n\t\t\t\t\t\t/WordSpa'
 b'cing [ .8 1.0 1.33 ]\n\t\t\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n\t\t'
 b'\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t\t\t\t/AutoLeading 1.2\n\t\t\t'
 b'\t\t\t/LeadingType 0\n\t\t\t\t\t\t/Hanging false\n\t\t\t\t\t\t/Burasagari'
 b' false\n\t\t\t\t\t\t/KinsokuOrder 0\n\t\t\t\t\t\t/EveryLineComposer false\n'
 b'\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t/Adjustments\n\t\t\t\t<<\n\t\t\t\t\t/Axi'
 b's [ 1.0 0.0 1.0 ]\n\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t]'
 b'\n\t\t\t/RunLengthArray [ 12 11 11 ]\n\t\t\t/IsJoinable 1\n\t\t>>\n\t\t/St'
 b'yleRun\n\t\t<<\n\t\t\t/DefaultRunData\n\t\t\t<<\n\t\t\t\t/StyleSheet\n\t\t\t'
 b'\t<<\n\t\t\t\t\t/StyleSheetData\n\t\t\t\t\t<<\n\t\t\t\t\t>>\n\t\t\t\t>>\n'
 b'\t\t\t>>\n\t\t\t/RunArray [\n\t\t\t<<\n\t\t\t\t/StyleSheet\n\t\t\t\t<<\n\t\t'
 b'\t\t\t/StyleSheetData\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Font 0\n\t\t\t\t\t\t/Font'
 b'Size 48.0\n\t\t\t\t\t\t/AutoKerning false\n\t\t\t\t\t\t/Kerning 0\n\t\t\t\t'
 b'\t\t/Language 14\n\t\t\t\t\t\t/FillColor\n\t\t\t\t\t\t<<\n\t\t\t\t\t\t\t/Typ'
 b'e 1\n\t\t\t\t\t\t\t/Values [ 1.0 .74547 .47001 .69903 ]\n\t\t\t\t\t\t>>'
 b'\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t<<\n\t\t\t\t/StyleSheet\n'
 b'\t\t\t\t<<\n\t\t\t\t\t/StyleSheetData\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Font '
 b'0\n\t\t\t\t\t\t/FontSize 48.0\n\t\t\t\t\t\t/AutoKerning true\n\t\t\t\t\t'
 b'\t/Kerning 0\n\t\t\t\t\t\t/Language 14\n\t\t\t\t\t\t/FillColor\n\t\t\t\t'
 b'\t\t<<\n\t\t\t\t\t\t\t/Type 1\n\t\t\t\t\t\t\t/Values [ 1.0 .74547 .47001 .'
 b'69903 ]\n\t\t\t\t\t\t>>\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t]\n\t'
 b'\t\t/RunLengthArray [ 1 33 ]\n\t\t\t/IsJoinable 2\n\t\t>>\n\t\t/GridInfo'
 b'\n\t\t<<\n\t\t\t/GridIsOn false\n\t\t\t/ShowGrid false\n\t\t\t/GridSize 18.'
 b'0\n\t\t\t/GridLeading 22.0\n\t\t\t/GridColor\n\t\t\t<<\n\t\t\t\t/Type 1\n\t'
 b'\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n\t\t\t>>\n\t\t\t/GridLeadingFillColo'
 b'r\n\t\t\t<<\n\t\t\t\t/Type 1\n\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n\t\t\t>'
 b'>\n\t\t\t/AlignLineHeightToGridFlags false\n\t\t>>\n\t\t/AntiAlias 4\n\t'
 b'\t/UseFractionalGlyphWidths true\n\t\t/Rendered\n\t\t<<\n\t\t\t/Version'
 b' 1\n\t\t\t/Shapes\n\t\t\t<<\n\t\t\t\t/WritingDirection 0\n\t\t\t\t/Childre'
 b'n [\n\t\t\t\t<<\n\t\t\t\t\t/ShapeType 0\n\t\t\t\t\t/Procession 0\n\t\t\t\t'
 b'\t/Lines\n\t\t\t\t\t<<\n\t\t\t\t\t\t/WritingDirection 0\n\t\t\t\t\t\t/Chi'
 b'ldren [ ]\n\t\t\t\t\t>>\n\t\t\t\t\t/Cookie\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Ph'
 b'otoshop\n\t\t\t\t\t\t<<\n\t\t\t\t\t\t\t/ShapeType 0\n\t\t\t\t\t\t\t/PointBa'
 b'se [ 0.0 0.0 ]\n\t\t\t\t\t\t\t/Base\n\t\t\t\t\t\t\t<<\n\t\t\t\t\t\t\t\t/S'
 b'hapeType 0\n\t\t\t\t\t\t\t\t/TransformPoint0 [ 1.0 0.0 ]\n\t\t\t\t\t\t\t\t'
 b'/TransformPoint1 [ 0.0 1.0 ]\n\t\t\t\t\t\t\t\t/TransformPoint2 [ 0.0 0.0 '
 b']\n\t\t\t\t\t\t\t>>\n\t\t\t\t\t\t>>\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t'
 b']\n\t\t\t>>\n\t\t>>\n\t>>\n\t/ResourceDict\n\t<<\n\t\t/KinsokuSet [\n\t\t<<'
 b'\n\t\t\t/Name (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00'
 b'K\x00i\x00n\x00s\x00o\x00k\x00u\x00H\x00a\x00r\x00d)\n\t\t\t/NoStart ('
 b'\xfe\xff0\x010\x02\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x01'
 b'0\xfc \x15 \x19 \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd'
 b'0\xfe0\x9d0\x9e0\x050A0C0E0G0I0c0\x830\x850\x870\x8e0\xa10\xa30\xa50\xa7'
 b'0\xa90\xc30\xe30\xe50\xe70\xee0\xf50\xf60\x9b0\x9c\x00?\x00!\x00\\)\x00'
 b']\x00}\x00,\x00.\x00:\x00;!\x03!\t\x00\xa2\xff\x05 0)\n\t\t\t/NoEnd'
 b' (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff[0\x080\n0\x0c0\x0e0\x10\x00\\'
 b'(\x00[\x00{\xff\xe5\xff\x04\x00\xa3\xff \x00\xa70\x12\xff\x03)\n\t\t\t/Kee'
 b'p (\xfe\xff \x15 %)\n\t\t\t/Hanging (\xfe\xff0\x010\x02\x00.\x00,)\n\t\t>>'
 b'\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p'
 b'\x00K\x00i\x00n\x00s\x00o\x00k\x00u\x00S\x00o\x00f\x00t)\n\t\t\t/NoStart '
 b'(\xfe\xff0\x010\x02\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff'
 b'\x01 \x19 \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0'
 b'\x9d0\x9e0\x05)\n\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff['
 b'0\x080\n0\x0c0\x0e0\x10)\n\t\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hangin'
 b'g (\xfe\xff0\x010\x02\x00.\x00,)\n\t\t>>\n\t\t]\n\t\t/MojiKumiSet [\n\t\t<'
 b'<\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h'
 b'\x00o\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t'
 b'\x001)\n\t\t>>\n\t\t<<\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o'
 b'\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m'
 b'\x00i\x00S\x00e\x00t\x002)\n\t\t>>\n\t\t<<\n\t\t\t/InternalName ('
 b'\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o\x00j'
 b'\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x003)\n\t\t>>\n\t\t<<\n\t\t\t/In'
 b'ternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M'
 b'\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x004)\n\t\t>>\n\t\t]'
 b'\n\t\t/TheNormalStyleSheet 0\n\t\t/TheNormalParagraphSheet 0\n\t\t/Paragr'
 b'aphSheetSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00m\x00'
 b'a\x00l\x00 \x00R\x00G\x00B)\n\t\t\t/DefaultStyleSheet 0\n\t\t\t/Properties\n'
 b'\t\t\t<<\n\t\t\t\t/Justification 0\n\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t'
 b'/StartIndent 0.0\n\t\t\t\t/EndIndent 0.0\n\t\t\t\t/SpaceBefore 0.0\n\t\t\t'
 b'\t/SpaceAfter 0.0\n\t\t\t\t/AutoHyphenate true\n\t\t\t\t/HyphenatedWordSize'
 b' 6\n\t\t\t\t/PreHyphen 2\n\t\t\t\t/PostHyphen 2\n\t\t\t\t/ConsecutiveHyphen'
 b's 8\n\t\t\t\t/Zone 36.0\n\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t/Let'
 b'terSpacing [ 0.0 0.0 0.0 ]\n\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t'
 b'\t/AutoLeading 1.2\n\t\t\t\t/LeadingType 0\n\t\t\t\t/Hanging false\n\t\t\t\t'
 b'/Burasagari false\n\t\t\t\t/KinsokuOrder 0\n\t\t\t\t/EveryLineComposer fal'
 b'se\n\t\t\t>>\n\t\t>>\n\t\t]\n\t\t/StyleSheetSet [\n\t\t<<\n\t\t\t/Name ('
 b'\xfe\xff\x00N\x00o\x00r\x00m\x00a\x00l\x00 \x00R\x00G\x00B)\n\t\t\t/StyleShe'
 b'etData\n\t\t\t<<\n\t\t\t\t/Font 2\n\t\t\t\t/FontSize 12.0\n\t\t\t\t/FauxBol'
 b'd false\n\t\t\t\t/FauxItalic false\n\t\t\t\t/AutoLeading true\n\t\t\t\t/Lea'
 b'ding 0.0\n\t\t\t\t/HorizontalScale 1.0\n\t\t\t\t/VerticalScale 1.0\n\t\t\t'
 b'\t/Tracking 0\n\t\t\t\t/AutoKerning true\n\t\t\t\t/Kerning 0\n\t\t\t\t/Basel'
 b'ineShift 0.0\n\t\t\t\t/FontCaps 0\n\t\t\t\t/FontBaseline 0\n\t\t\t\t/Underl'
 b'ine false\n\t\t\t\t/Strikethrough false\n\t\t\t\t/Ligatures true\n\t\t\t\t/'
 b'DLigatures false\n\t\t\t\t/BaselineDirection 2\n\t\t\t\t/Tsume 0.0\n\t\t\t'
 b'\t/StyleRunAlignment 2\n\t\t\t\t/Language 0\n\t\t\t\t/NoBreak false\n\t\t\t'
 b'\t/FillColor\n\t\t\t\t<<\n\t\t\t\t\t/Type 1\n\t\t\t\t\t/Values [ 1.0 0.0 0'
 b'.0 0.0 ]\n\t\t\t\t>>\n\t\t\t\t/StrokeColor\n\t\t\t\t<<\n\t\t\t\t\t/Type 1'
 b'\n\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t\t/FillFlag t'
 b'rue\n\t\t\t\t/StrokeFlag false\n\t\t\t\t/FillFirst true\n\t\t\t\t/YUnderlin'
 b'e 1\n\t\t\t\t/OutlineWidth 1.0\n\t\t\t\t/CharacterDirection 0\n\t\t\t\t/Hin'
 b'diNumbers false\n\t\t\t\t/Kashida 1\n\t\t\t\t/DiacriticPos 2\n\t\t\t>>'
 b'\n\t\t>>\n\t\t]\n\t\t/FontSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00C\x00e'
 b'\x00n\x00t\x00u\x00r\x00y\x00G\x00o\x00t\x00h\x00i\x00c\x00-\x00B\x00o'
 b'\x00l\x00d)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic 0\n\t\t'
 b'>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00A\x00d\x00o\x00b\x00e\x00I\x00n\x00v'
 b'\x00i\x00s\x00F\x00o\x00n\x00t)\n\t\t\t/Script 0\n\t\t\t/FontType 0\n\t\t'
 b'\t/Synthetic 0\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00T\x00a\x00h\x00o'
 b'\x00m\x00a)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic 0\n\t\t'
 b'>>\n\t\t]\n\t\t/SuperscriptSize .583\n\t\t/SuperscriptPosition .333\n\t\t/Su'
 b'bscriptSize .583\n\t\t/SubscriptPosition .333\n\t\t/SmallCapSize .7\n\t>'
 b'>\n\t/DocumentResources\n\t<<\n\t\t/KinsokuSet [\n\t\t<<\n\t\t\t/Name '
 b'(\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00'
 b's\x00o\x00k\x00u\x00H\x00a\x00r\x00d)\n\t\t\t/NoStart (\xfe\xff0\x010\x02'
 b'\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x010\xfc \x15 \x19'
 b' \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0\x9d0\x9e0\x050A'
 b'0C0E0G0I0c0\x830\x850\x870\x8e0\xa10\xa30\xa50\xa70\xa90\xc30\xe30\xe50\xe7'
 b'0\xee0\xf50\xf60\x9b0\x9c\x00?\x00!\x00\\)\x00]\x00}\x00,\x00.\x00:\x00'
 b';!\x03!\t\x00\xa2\xff\x05 0)\n\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x08'
 b'0\x14\xff;\xff[0\x080\n0\x0c0\x0e0\x10\x00\\(\x00[\x00{\xff\xe5\xff\x04\x00'
 b'\xa3\xff \x00\xa70\x12\xff\x03)\n\t\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Han'
 b'ging (\xfe\xff0\x010\x02\x00.\x00,)\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff'
 b'\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00s\x00o'
 b'\x00k\x00u\x00S\x00o\x00f\x00t)\n\t\t\t/NoStart (\xfe\xff0\x010'
 b'\x02\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x01 \x19 \x1d\xff'
 b'\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0\x9d0\x9e0\x05)\n\t\t\t/NoE'
 b'nd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff[0\x080\n0\x0c0\x0e0\x10)\n\t\t'
 b'\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hanging (\xfe\xff0\x010\x02\x00.\x00,)'
 b'\n\t\t>>\n\t\t]\n\t\t/MojiKumiSet [\n\t\t<<\n\t\t\t/InternalName (\xfe\xff'
 b'\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o\x00j\x00i'
 b'\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x001)\n\t\t>>\n\t\t<<\n\t\t\t/Internal'
 b'Name (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o'
 b'\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x002)\n\t\t>>\n\t\t<<\n\t\t\t/'
 b'InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006'
 b'\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x003)\n\t\t>>\n\t'
 b'\t<<\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o'
 b'\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x004'
 b')\n\t\t>>\n\t\t]\n\t\t/TheNormalStyleSheet 0\n\t\t/TheNormalParagraphSheet 0'
 b'\n\t\t/ParagraphSheetSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00'
 b'm\x00a\x00l\x00 \x00R\x00G\x00B)\n\t\t\t/DefaultStyleSheet 0\n\t\t\t/Prope'
 b'rties\n\t\t\t<<\n\t\t\t\t/Justification 0\n\t\t\t\t/FirstLineIndent 0.'
 b'0\n\t\t\t\t/StartIndent 0.0\n\t\t\t\t/EndIndent 0.0\n\t\t\t\t/SpaceBefore 0'
 b'.0\n\t\t\t\t/SpaceAfter 0.0\n\t\t\t\t/AutoHyphenate true\n\t\t\t\t/Hyphenat'
 b'edWordSize 6\n\t\t\t\t/PreHyphen 2\n\t\t\t\t/PostHyphen 2\n\t\t\t\t/Consecu'
 b'tiveHyphens 8\n\t\t\t\t/Zone 36.0\n\t\t\t\t/WordSpacing [ .8 1.0 1.33 '
 b']\n\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n\t\t\t\t/GlyphSpacing [ 1.0 1.0'
 b' 1.0 ]\n\t\t\t\t/AutoLeading 1.2\n\t\t\t\t/LeadingType 0\n\t\t\t\t/Hanging '
 b'false\n\t\t\t\t/Burasagari false\n\t\t\t\t/KinsokuOrder 0\n\t\t\t\t/EveryLi'
 b'neComposer false\n\t\t\t>>\n\t\t>>\n\t\t]\n\t\t/StyleSheetSet [\n\t\t<<\n'
 b'\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00m\x00a\x00l\x00 \x00R\x00G\x00B'
 b')\n\t\t\t/StyleSheetData\n\t\t\t<<\n\t\t\t\t/Font 2\n\t\t\t\t/FontSize 12.'
 b'0\n\t\t\t\t/FauxBold false\n\t\t\t\t/FauxItalic false\n\t\t\t\t/AutoLeading'
 b' true\n\t\t\t\t/Leading 0.0\n\t\t\t\t/HorizontalScale 1.0\n\t\t\t\t/Vertica'
 b'lScale 1.0\n\t\t\t\t/Tracking 0\n\t\t\t\t/AutoKerning true\n\t\t\t\t/Kernin'
 b'g 0\n\t\t\t\t/BaselineShift 0.0\n\t\t\t\t/FontCaps 0\n\t\t\t\t/FontBaseline'
 b' 0\n\t\t\t\t/Underline false\n\t\t\t\t/Strikethrough false\n\t\t\t\t/Ligatu'
 b'res true\n\t\t\t\t/DLigatures false\n\t\t\t\t/BaselineDirection 2\n\t\t\t\t'
 b'/Tsume 0.0\n\t\t\t\t/StyleRunAlignment 2\n\t\t\t\t/Language 0\n\t\t\t\t/NoB'
 b'reak false\n\t\t\t\t/FillColor\n\t\t\t\t<<\n\t\t\t\t\t/Type 1\n\t\t\t\t\t/'
 b'Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t\t/StrokeColor\n\t\t\t\t<'
 b'<\n\t\t\t\t\t/Type 1\n\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t'
 b'\t\t\t/FillFlag true\n\t\t\t\t/StrokeFlag false\n\t\t\t\t/FillFirst true\n'
 b'\t\t\t\t/YUnderline 1\n\t\t\t\t/OutlineWidth 1.0\n\t\t\t\t/CharacterDirect'
 b'ion 0\n\t\t\t\t/HindiNumbers false\n\t\t\t\t/Kashida 1\n\t\t\t\t/DiacriticP'
 b'os 2\n\t\t\t>>\n\t\t>>\n\t\t]\n\t\t/FontSet [\n\t\t<<\n\t\t\t/Name ('
 b'\xfe\xff\x00C\x00e\x00n\x00t\x00u\x00r\x00y\x00G\x00o\x00t\x00h\x00i\x00c'
 b'\x00-\x00B\x00o\x00l\x00d)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synt'
 b'hetic 0\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00A\x00d\x00o\x00b\x00e'
 b'\x00I\x00n\x00v\x00i\x00s\x00F\x00o\x00n\x00t)\n\t\t\t/Script 0\n\t\t\t/Fon'
 b'tType 0\n\t\t\t/Synthetic 0\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00T'
 b'\x00a\x00h\x00o\x00m\x00a)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synt'
 b'hetic 0\n\t\t>>\n\t\t]\n\t\t/SuperscriptSize .583\n\t\t/SuperscriptPosition '
 b'.333\n\t\t/SubscriptSize .583\n\t\t/SubscriptPosition .333\n\t\t/SmallCap'
 b'Size .7\n\t>>\n>>')

psd_text_template = (
b'\x00\x01?\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
b'\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00@f`\x00\x00\x00\x00\x00@'
b'v\x80\x00\x00\x00\x00\x00\x002\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00'
b'\x00\x00\x00TxLr\x00\x00\x00\x08\x00\x00\x00\x00Txt TEXT\x00\x00\x00"\x00t'
b'\x00e\x00s\x00t\x00 \x00s\x00t\x00r\x00i\x00n\x00g\x00\r\x00o\x00t\x00h\x00e'
b'\x00r\x00 \x00l\x00i\x00n\x00e\x00\r\x00t\x00h\x00i\x00r\x00d\x00 \x00l\x00i'
b'\x00n\x00e\x00\x00\x00\x00\x00\x0ctextGriddingenum\x00\x00\x00\x0ctextGriddin'
b'g\x00\x00\x00\x00None\x00\x00\x00\x00Orntenum\x00\x00\x00\x00Ornt\x00\x00\x00'
b'\x00Hrzn\x00\x00\x00\x00AntAenum\x00\x00\x00\x00Annt\x00\x00\x00\x0eantiAlias'
b'Sharp\x00\x00\x00\x06boundsObjc\x00\x00\x00\x01\x00\x00\x00\x00\x00\x06bounds'
b'\x00\x00\x00\x04\x00\x00\x00\x00LeftUntF#Pnt\xc0[\x1d@\x00\x00\x00\x00\x00'
b'\x00\x00\x00Top UntF#Pnt\xc0D\x9d\xa8\x00\x00\x00\x00\x00\x00\x00\x00RghtUntF'
b'#Pnt@[\x1d@\x00\x00\x00\x00\x00\x00\x00\x00BtomUntF#Pnt@`=f\x80\x00\x00\x00'
b'\x00\x00\x00\x0bboundingBoxObjc\x00\x00\x00\x01\x00\x00\x00\x00\x00\x0bboundi'
b'ngBox\x00\x00\x00\x04\x00\x00\x00\x00LeftUntF#Pnt\xc0Z\xdd@\x00\x00\x00\x00'
b'\x00\x00\x00\x00Top UntF#Pnt\xc0A\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00Rght'
b'UntF#Pnt@Z\x90\xc0\x00\x00\x00\x00\x00\x00\x00\x00BtomUntF#Pnt@\\\xcc\xcd\x00'
b'\x00\x00\x00\x00\x00\x00\tTextIndexlong\x00\x00\x00\x00\x00\x00\x00\nEngineDa'
b'tatdta\x00\x00&\x86\n\n<<\n\t/EngineDict\n\t<<\n\t\t/Editor\n\t\t<<\n\t\t\t/T'
b'ext (\xfe\xff\x00t\x00e\x00s\x00t\x00 \x00s\x00t\x00r\x00i\x00n\x00g\x00\r'
b'\x00o\x00t\x00h\x00e\x00r\x00 \x00l\x00i\x00n\x00e\x00\r\x00t\x00h\x00i\x00r'
b'\x00d\x00 \x00l\x00i\x00n\x00e\x00\r)\n\t\t>>\n\t\t/ParagraphRun\n\t\t<<\n\t'
b'\t\t/DefaultRunData\n\t\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t'
b'\t/DefaultStyleSheet 0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t>>\n\t'
b'\t\t\t>>\n\t\t\t\t/Adjustments\n\t\t\t\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n'
b'\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t/RunArray [\n\t\t\t<<'
b'\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/DefaultStyleSheet 0\n\t\t\t'
b'\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justification 2\n\t\t\t\t\t\t/Fir'
b'stLineIndent 0.0\n\t\t\t\t\t\t/StartIndent 0.0\n\t\t\t\t\t\t/EndIndent 0.0\n'
b'\t\t\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t/SpaceAfter 0.0\n\t\t\t\t\t\t/AutoH'
b'yphenate true\n\t\t\t\t\t\t/HyphenatedWordSize 6\n\t\t\t\t\t\t/PreHyphen 2\n'
b'\t\t\t\t\t\t/PostHyphen 2\n\t\t\t\t\t\t/ConsecutiveHyphens 8\n\t\t\t\t\t\t/Zo'
b'ne 36.0\n\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t\t\t/LetterSpacing'
b' [ 0.0 0.0 0.0 ]\n\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t\t\t\t/Aut'
b'oLeading 1.2\n\t\t\t\t\t\t/LeadingType 0\n\t\t\t\t\t\t/Hanging false\n\t\t\t'
b'\t\t\t/Burasagari false\n\t\t\t\t\t\t/KinsokuOrder 0\n\t\t\t\t\t\t/EveryLineC'
b'omposer false\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t/Adjustments\n\t\t\t\t<<\n\t'
b'\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t>'
b'>\n\t\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/DefaultStyleShee'
b't 0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justification 2\n\t\t'
b'\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t\t\t/StartIndent 0.0\n\t\t\t\t\t\t/EndI'
b'ndent 0.0\n\t\t\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t/SpaceAfter 0.0\n\t\t\t'
b'\t\t\t/AutoHyphenate true\n\t\t\t\t\t\t/HyphenatedWordSize 6\n\t\t\t\t\t\t/Pr'
b'eHyphen 2\n\t\t\t\t\t\t/PostHyphen 2\n\t\t\t\t\t\t/ConsecutiveHyphens 8\n\t\t'
b'\t\t\t\t/Zone 36.0\n\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t\t\t/Le'
b'tterSpacing [ 0.0 0.0 0.0 ]\n\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t'
b'\t\t\t\t/AutoLeading 1.2\n\t\t\t\t\t\t/LeadingType 0\n\t\t\t\t\t\t/Hanging fa'
b'lse\n\t\t\t\t\t\t/Burasagari false\n\t\t\t\t\t\t/KinsokuOrder 0\n\t\t\t\t\t\t'
b'/EveryLineComposer false\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t/Adjustments\n\t'
b'\t\t\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n\t\t\t\t\t/XY [ 0.0 0.0 ]\n\t\t\t'
b'\t>>\n\t\t\t>>\n\t\t\t<<\n\t\t\t\t/ParagraphSheet\n\t\t\t\t<<\n\t\t\t\t\t/Def'
b'aultStyleSheet 0\n\t\t\t\t\t/Properties\n\t\t\t\t\t<<\n\t\t\t\t\t\t/Justifica'
b'tion 2\n\t\t\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t\t\t/StartIndent 0.0\n\t\t'
b'\t\t\t\t/EndIndent 0.0\n\t\t\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t\t\t/SpaceAfter'
b' 0.0\n\t\t\t\t\t\t/AutoHyphenate true\n\t\t\t\t\t\t/HyphenatedWordSize 6\n\t'
b'\t\t\t\t\t/PreHyphen 2\n\t\t\t\t\t\t/PostHyphen 2\n\t\t\t\t\t\t/ConsecutiveHy'
b'phens 8\n\t\t\t\t\t\t/Zone 36.0\n\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t'
b'\t\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0'
b' 1.0 ]\n\t\t\t\t\t\t/AutoLeading 1.2\n\t\t\t\t\t\t/LeadingType 0\n\t\t\t\t\t'
b'\t/Hanging false\n\t\t\t\t\t\t/Burasagari false\n\t\t\t\t\t\t/KinsokuOrder 0'
b'\n\t\t\t\t\t\t/EveryLineComposer false\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t\t/Ad'
b'justments\n\t\t\t\t<<\n\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n\t\t\t\t\t/XY [ 0.0 0'
b'.0 ]\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t]\n\t\t\t/RunLengthArray [ 12 11 11 ]\n\t\t'
b'\t/IsJoinable 1\n\t\t>>\n\t\t/StyleRun\n\t\t<<\n\t\t\t/DefaultRunData\n\t\t\t'
b'<<\n\t\t\t\t/StyleSheet\n\t\t\t\t<<\n\t\t\t\t\t/StyleSheetData\n\t\t\t\t\t<<'
b'\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t/RunArray [\n\t\t\t<<\n\t\t\t\t/S'
b'tyleSheet\n\t\t\t\t<<\n\t\t\t\t\t/StyleSheetData\n\t\t\t\t\t<<\n\t\t\t\t\t\t/'
b'Font 0\n\t\t\t\t\t\t/FontSize 48.0\n\t\t\t\t\t\t/AutoKerning false\n\t\t\t\t'
b'\t\t/Kerning 0\n\t\t\t\t\t\t/Language 14\n\t\t\t\t\t\t/FillColor\n\t\t\t\t\t'
b'\t<<\n\t\t\t\t\t\t\t/Type 1\n\t\t\t\t\t\t\t/Values [ 1.0 .74547 .47001 .69903'
b' ]\n\t\t\t\t\t\t>>\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t<<\n\t\t\t\t/St'
b'yleSheet\n\t\t\t\t<<\n\t\t\t\t\t/StyleSheetData\n\t\t\t\t\t<<\n\t\t\t\t\t\t/F'
b'ont 0\n\t\t\t\t\t\t/FontSize 48.0\n\t\t\t\t\t\t/AutoKerning true\n\t\t\t\t\t'
b'\t/Kerning 0\n\t\t\t\t\t\t/Language 14\n\t\t\t\t\t\t/FillColor\n\t\t\t\t\t\t<'
b'<\n\t\t\t\t\t\t\t/Type 1\n\t\t\t\t\t\t\t/Values [ 1.0 .74547 .47001 .69903 ]'
b'\n\t\t\t\t\t\t>>\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t\t\t>>\n\t\t\t]\n\t\t\t/RunLeng'
b'thArray [ 1 33 ]\n\t\t\t/IsJoinable 2\n\t\t>>\n\t\t/GridInfo\n\t\t<<\n\t\t\t/'
b'GridIsOn false\n\t\t\t/ShowGrid false\n\t\t\t/GridSize 18.0\n\t\t\t/GridLeadi'
b'ng 22.0\n\t\t\t/GridColor\n\t\t\t<<\n\t\t\t\t/Type 1\n\t\t\t\t/Values [ 0.0 0'
b'.0 0.0 1.0 ]\n\t\t\t>>\n\t\t\t/GridLeadingFillColor\n\t\t\t<<\n\t\t\t\t/Type '
b'1\n\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n\t\t\t>>\n\t\t\t/AlignLineHeightToGri'
b'dFlags false\n\t\t>>\n\t\t/AntiAlias 4\n\t\t/UseFractionalGlyphWidths true\n'
b'\t\t/Rendered\n\t\t<<\n\t\t\t/Version 1\n\t\t\t/Shapes\n\t\t\t<<\n\t\t\t\t/Wr'
b'itingDirection 0\n\t\t\t\t/Children [\n\t\t\t\t<<\n\t\t\t\t\t/ShapeType 0\n\t'
b'\t\t\t\t/Procession 0\n\t\t\t\t\t/Lines\n\t\t\t\t\t<<\n\t\t\t\t\t\t/WritingDi'
b'rection 0\n\t\t\t\t\t\t/Children [ ]\n\t\t\t\t\t>>\n\t\t\t\t\t/Cookie\n\t\t\t'
b'\t\t<<\n\t\t\t\t\t\t/Photoshop\n\t\t\t\t\t\t<<\n\t\t\t\t\t\t\t/ShapeType 0\n'
b'\t\t\t\t\t\t\t/PointBase [ 0.0 0.0 ]\n\t\t\t\t\t\t\t/Base\n\t\t\t\t\t\t\t<<\n'
b'\t\t\t\t\t\t\t\t/ShapeType 0\n\t\t\t\t\t\t\t\t/TransformPoint0 [ 1.0 0.0 ]\n'
b'\t\t\t\t\t\t\t\t/TransformPoint1 [ 0.0 1.0 ]\n\t\t\t\t\t\t\t\t/TransformPoint'
b'2 [ 0.0 0.0 ]\n\t\t\t\t\t\t\t>>\n\t\t\t\t\t\t>>\n\t\t\t\t\t>>\n\t\t\t\t>>\n\t'
b'\t\t\t]\n\t\t\t>>\n\t\t>>\n\t>>\n\t/ResourceDict\n\t<<\n\t\t/KinsokuSet [\n\t'
b'\t<<\n\t\t\t/Name (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K'
b'\x00i\x00n\x00s\x00o\x00k\x00u\x00H\x00a\x00r\x00d)\n\t\t\t/NoStart (\xfe\xff'
b'0\x010\x02\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x010\xfc \x15 '
b'\x19 \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0\x9d0\x9e0\x050'
b'A0C0E0G0I0c0\x830\x850\x870\x8e0\xa10\xa30\xa50\xa70\xa90\xc30\xe30\xe50\xe70'
b'\xee0\xf50\xf60\x9b0\x9c\x00?\x00!\x00\\)\x00]\x00}\x00,\x00.\x00:\x00;!\x03!'
b'\t\x00\xa2\xff\x05 0)\n\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff'
b'[0\x080\n0\x0c0\x0e0\x10\x00\\(\x00[\x00{\xff\xe5\xff\x04\x00\xa3\xff \x00'
b'\xa70\x12\xff\x03)\n\t\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hanging (\xfe\xff0'
b'\x010\x02\x00.\x00,)\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00P\x00h\x00o'
b'\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00s\x00o\x00k\x00u\x00S\x00o'
b'\x00f\x00t)\n\t\t\t/NoStart (\xfe\xff0\x010\x02\xff\x0c\xff\x0e0\xfb\xff\x1a'
b'\xff\x1b\xff\x1f\xff\x01 \x19 \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110'
b'\xfd0\xfe0\x9d0\x9e0\x05)\n\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;'
b'\xff[0\x080\n0\x0c0\x0e0\x10)\n\t\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hanging '
b'(\xfe\xff0\x010\x02\x00.\x00,)\n\t\t>>\n\t\t]\n\t\t/MojiKumiSet [\n\t\t<<\n\t'
b'\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006'
b'\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x001)\n\t\t>>\n\t\t<<'
b'\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p'
b'\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x002)\n\t\t>>\n'
b'\t\t<<\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o'
b'\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x003)\n\t\t'
b'>>\n\t\t<<\n\t\t\t/InternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h'
b'\x00o\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x004)'
b'\n\t\t>>\n\t\t]\n\t\t/TheNormalStyleSheet 0\n\t\t/TheNormalParagraphSheet 0\n'
b'\t\t/ParagraphSheetSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00m'
b'\x00a\x00l\x00 \x00R\x00G\x00B)\n\t\t\t/DefaultStyleSheet 0\n\t\t\t/Propertie'
b's\n\t\t\t<<\n\t\t\t\t/Justification 0\n\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t'
b'/StartIndent 0.0\n\t\t\t\t/EndIndent 0.0\n\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t/'
b'SpaceAfter 0.0\n\t\t\t\t/AutoHyphenate true\n\t\t\t\t/HyphenatedWordSize 6\n'
b'\t\t\t\t/PreHyphen 2\n\t\t\t\t/PostHyphen 2\n\t\t\t\t/ConsecutiveHyphens 8\n'
b'\t\t\t\t/Zone 36.0\n\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t/LetterSpac'
b'ing [ 0.0 0.0 0.0 ]\n\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t\t/AutoLead'
b'ing 1.2\n\t\t\t\t/LeadingType 0\n\t\t\t\t/Hanging false\n\t\t\t\t/Burasagari '
b'false\n\t\t\t\t/KinsokuOrder 0\n\t\t\t\t/EveryLineComposer false\n\t\t\t>>\n'
b'\t\t>>\n\t\t]\n\t\t/StyleSheetSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o'
b'\x00r\x00m\x00a\x00l\x00 \x00R\x00G\x00B)\n\t\t\t/StyleSheetData\n\t\t\t<<\n'
b'\t\t\t\t/Font 2\n\t\t\t\t/FontSize 12.0\n\t\t\t\t/FauxBold false\n\t\t\t\t/Fa'
b'uxItalic false\n\t\t\t\t/AutoLeading true\n\t\t\t\t/Leading 0.0\n\t\t\t\t/Hor'
b'izontalScale 1.0\n\t\t\t\t/VerticalScale 1.0\n\t\t\t\t/Tracking 0\n\t\t\t\t/A'
b'utoKerning true\n\t\t\t\t/Kerning 0\n\t\t\t\t/BaselineShift 0.0\n\t\t\t\t/Fon'
b'tCaps 0\n\t\t\t\t/FontBaseline 0\n\t\t\t\t/Underline false\n\t\t\t\t/Striketh'
b'rough false\n\t\t\t\t/Ligatures true\n\t\t\t\t/DLigatures false\n\t\t\t\t/Bas'
b'elineDirection 2\n\t\t\t\t/Tsume 0.0\n\t\t\t\t/StyleRunAlignment 2\n\t\t\t\t/'
b'Language 0\n\t\t\t\t/NoBreak false\n\t\t\t\t/FillColor\n\t\t\t\t<<\n\t\t\t\t'
b'\t/Type 1\n\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t\t/Stroke'
b'Color\n\t\t\t\t<<\n\t\t\t\t\t/Type 1\n\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n'
b'\t\t\t\t>>\n\t\t\t\t/FillFlag true\n\t\t\t\t/StrokeFlag false\n\t\t\t\t/FillF'
b'irst true\n\t\t\t\t/YUnderline 1\n\t\t\t\t/OutlineWidth 1.0\n\t\t\t\t/Charact'
b'erDirection 0\n\t\t\t\t/HindiNumbers false\n\t\t\t\t/Kashida 1\n\t\t\t\t/Diac'
b'riticPos 2\n\t\t\t>>\n\t\t>>\n\t\t]\n\t\t/FontSet [\n\t\t<<\n\t\t\t/Name ('
b'\xfe\xff\x00C\x00e\x00n\x00t\x00u\x00r\x00y\x00G\x00o\x00t\x00h\x00i\x00c\x00'
b'-\x00B\x00o\x00l\x00d)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic '
b'0\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00A\x00d\x00o\x00b\x00e\x00I\x00n'
b'\x00v\x00i\x00s\x00F\x00o\x00n\x00t)\n\t\t\t/Script 0\n\t\t\t/FontType 0\n\t'
b'\t\t/Synthetic 0\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00T\x00a\x00h\x00o'
b'\x00m\x00a)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic 0\n\t\t>>\n'
b'\t\t]\n\t\t/SuperscriptSize .583\n\t\t/SuperscriptPosition .333\n\t\t/Subscri'
b'ptSize .583\n\t\t/SubscriptPosition .333\n\t\t/SmallCapSize .7\n\t>>\n\t/Docu'
b'mentResources\n\t<<\n\t\t/KinsokuSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00P'
b'\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00s\x00o\x00k\x00u'
b'\x00H\x00a\x00r\x00d)\n\t\t\t/NoStart (\xfe\xff0\x010\x02\xff\x0c\xff\x0e0'
b'\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x010\xfc \x15 \x19 \x1d\xff\t0\x15\xff=\xff]'
b'0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0\x9d0\x9e0\x050A0C0E0G0I0c0\x830\x850\x870'
b'\x8e0\xa10\xa30\xa50\xa70\xa90\xc30\xe30\xe50\xe70\xee0\xf50\xf60\x9b0\x9c'
b'\x00?\x00!\x00\\)\x00]\x00}\x00,\x00.\x00:\x00;!\x03!\t\x00\xa2\xff\x05 0)\n'
b'\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff[0\x080\n0\x0c0\x0e0'
b'\x10\x00\\(\x00[\x00{\xff\xe5\xff\x04\x00\xa3\xff \x00\xa70\x12\xff\x03)\n\t'
b'\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hanging (\xfe\xff0\x010\x02\x00.\x00,)\n'
b'\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o'
b'\x00p\x00K\x00i\x00n\x00s\x00o\x00k\x00u\x00S\x00o\x00f\x00t)\n\t\t\t/NoStart'
b' (\xfe\xff0\x010\x02\xff\x0c\xff\x0e0\xfb\xff\x1a\xff\x1b\xff\x1f\xff\x01 '
b'\x19 \x1d\xff\t0\x15\xff=\xff]0\t0\x0b0\r0\x0f0\x110\xfd0\xfe0\x9d0\x9e0\x05)'
b'\n\t\t\t/NoEnd (\xfe\xff \x18 \x1c\xff\x080\x14\xff;\xff[0\x080\n0\x0c0\x0e0'
b'\x10)\n\t\t\t/Keep (\xfe\xff \x15 %)\n\t\t\t/Hanging (\xfe\xff0\x010\x02\x00.'
b'\x00,)\n\t\t>>\n\t\t]\n\t\t/MojiKumiSet [\n\t\t<<\n\t\t\t/InternalName (\xfe'
b'\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o\x00j\x00i'
b'\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x001)\n\t\t>>\n\t\t<<\n\t\t\t/InternalNam'
b'e (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o\x00j'
b'\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x002)\n\t\t>>\n\t\t<<\n\t\t\t/Intern'
b'alName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M\x00o'
b'\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x003)\n\t\t>>\n\t\t<<\n\t\t\t/I'
b'nternalName (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00M'
b'\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x004)\n\t\t>>\n\t\t]\n\t\t'
b'/TheNormalStyleSheet 0\n\t\t/TheNormalParagraphSheet 0\n\t\t/ParagraphSheetSe'
b't [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00m\x00a\x00l\x00 \x00R'
b'\x00G\x00B)\n\t\t\t/DefaultStyleSheet 0\n\t\t\t/Properties\n\t\t\t<<\n\t\t\t'
b'\t/Justification 0\n\t\t\t\t/FirstLineIndent 0.0\n\t\t\t\t/StartIndent 0.0\n'
b'\t\t\t\t/EndIndent 0.0\n\t\t\t\t/SpaceBefore 0.0\n\t\t\t\t/SpaceAfter 0.0\n\t'
b'\t\t\t/AutoHyphenate true\n\t\t\t\t/HyphenatedWordSize 6\n\t\t\t\t/PreHyphen '
b'2\n\t\t\t\t/PostHyphen 2\n\t\t\t\t/ConsecutiveHyphens 8\n\t\t\t\t/Zone 36.0\n'
b'\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]'
b'\n\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n\t\t\t\t/AutoLeading 1.2\n\t\t\t\t/L'
b'eadingType 0\n\t\t\t\t/Hanging false\n\t\t\t\t/Burasagari false\n\t\t\t\t/Kin'
b'sokuOrder 0\n\t\t\t\t/EveryLineComposer false\n\t\t\t>>\n\t\t>>\n\t\t]\n\t\t/'
b'StyleSheetSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00N\x00o\x00r\x00m\x00a\x00l'
b'\x00 \x00R\x00G\x00B)\n\t\t\t/StyleSheetData\n\t\t\t<<\n\t\t\t\t/Font 2\n\t\t'
b'\t\t/FontSize 12.0\n\t\t\t\t/FauxBold false\n\t\t\t\t/FauxItalic false\n\t\t'
b'\t\t/AutoLeading true\n\t\t\t\t/Leading 0.0\n\t\t\t\t/HorizontalScale 1.0\n\t'
b'\t\t\t/VerticalScale 1.0\n\t\t\t\t/Tracking 0\n\t\t\t\t/AutoKerning true\n\t'
b'\t\t\t/Kerning 0\n\t\t\t\t/BaselineShift 0.0\n\t\t\t\t/FontCaps 0\n\t\t\t\t/F'
b'ontBaseline 0\n\t\t\t\t/Underline false\n\t\t\t\t/Strikethrough false\n\t\t\t'
b'\t/Ligatures true\n\t\t\t\t/DLigatures false\n\t\t\t\t/BaselineDirection 2\n'
b'\t\t\t\t/Tsume 0.0\n\t\t\t\t/StyleRunAlignment 2\n\t\t\t\t/Language 0\n\t\t\t'
b'\t/NoBreak false\n\t\t\t\t/FillColor\n\t\t\t\t<<\n\t\t\t\t\t/Type 1\n\t\t\t\t'
b'\t/Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t\t/StrokeColor\n\t\t\t\t<<\n'
b'\t\t\t\t\t/Type 1\n\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n\t\t\t\t>>\n\t\t\t'
b'\t/FillFlag true\n\t\t\t\t/StrokeFlag false\n\t\t\t\t/FillFirst true\n\t\t\t'
b'\t/YUnderline 1\n\t\t\t\t/OutlineWidth 1.0\n\t\t\t\t/CharacterDirection 0\n\t'
b'\t\t\t/HindiNumbers false\n\t\t\t\t/Kashida 1\n\t\t\t\t/DiacriticPos 2\n\t\t'
b'\t>>\n\t\t>>\n\t\t]\n\t\t/FontSet [\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00C\x00e'
b'\x00n\x00t\x00u\x00r\x00y\x00G\x00o\x00t\x00h\x00i\x00c\x00-\x00B\x00o\x00l'
b'\x00d)\n\t\t\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic 0\n\t\t>>\n\t\t<'
b'<\n\t\t\t/Name (\xfe\xff\x00A\x00d\x00o\x00b\x00e\x00I\x00n\x00v\x00i\x00s'
b'\x00F\x00o\x00n\x00t)\n\t\t\t/Script 0\n\t\t\t/FontType 0\n\t\t\t/Synthetic 0'
b'\n\t\t>>\n\t\t<<\n\t\t\t/Name (\xfe\xff\x00T\x00a\x00h\x00o\x00m\x00a)\n\t\t'
b'\t/Script 0\n\t\t\t/FontType 1\n\t\t\t/Synthetic 0\n\t\t>>\n\t\t]\n\t\t/Super'
b'scriptSize .583\n\t\t/SuperscriptPosition .333\n\t\t/SubscriptSize .583\n\t\t'
b'/SubscriptPosition .333\n\t\t/SmallCapSize .7\n\t>>\n>>\x00\x01\x00\x00\x00'
b'\x10\x00\x00\x00\x01\x00\x00\x00\x00\x00\x04warp\x00\x00\x00\x05\x00\x00\x00'
b'\twarpStyleenum\x00\x00\x00\twarpStyle\x00\x00\x00\x08warpNone\x00\x00\x00\tw'
b'arpValuedoub\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0fwarpPerspectivedo'
b'ub\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14warpPerspectiveOtherdoub'
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\nwarpRotateenum\x00\x00\x00\x00O'
b'rnt\x00\x00\x00\x00Hrzn\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
b'\x00\x00\x00\x00'
)

global_text_tag =  (
b'\x00\x00\x00\x008BIMPatt\x00\x00\x00\x008BIMTxt2\x00\x00,{ /98 << /0 7 >> /0 '
b'<< /1 << /0 [ << /0 << /99 /CoolTypeFont /0 << /0 (\xfe\xff\x00C\x00e\x00n'
b'\x00t\x00u\x00r\x00y\x00G\x00o\x00t\x00h\x00i\x00c\x00-\x00B\x00o\x00l\x00d) '
b'/2 1 >> >> >> << /0 << /99 /CoolTypeFont /0 << /0 (\xfe\xff\x00A\x00d\x00o'
b'\x00b\x00e\x00I\x00n\x00v\x00i\x00s\x00F\x00o\x00n\x00t) /2 0 >> >> >> << /0 '
b'<< /99 /CoolTypeFont /0 << /0 (\xfe\xff\x00T\x00a\x00h\x00o\x00m\x00a) /2 1 >'
b'> >> >> << /0 << /99 /CoolTypeFont /0 << /0 (\xfe\xff\x00T\x00i\x00m\x00e\x00'
b's\x00N\x00e\x00w\x00R\x00o\x00m\x00a\x00n\x00P\x00S\x00M\x00T) /2 1 >> >> >> '
b'] >> /2 << /0 [ << /0 << /0 (\xfe\xff) >> >> ] /1 [ << /0 0 >> ] >> /3 << /0 '
b'[ << /0 << /0 (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006\x00'
b'M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x004) /5 << /0 0 /3 2 >> '
b'>> >> << /0 << /0 (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x006'
b'\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x003) /5 << /0 0 /3 4'
b' >> >> >> << /0 << /0 (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p'
b'\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x002) /5 << /0 0'
b' /3 3 >> >> >> << /0 << /0 (\xfe\xff\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o'
b'\x00p\x006\x00M\x00o\x00j\x00i\x00K\x00u\x00m\x00i\x00S\x00e\x00t\x001) /5 <<'
b' /0 0 /3 1 >> >> >> << /0 << /0 (\xfe\xff\x00Y\x00a\x00k\x00u\x00m\x00o\x00n'
b'\x00o\x00H\x00a\x00n\x00k\x00a\x00k\x00u) /5 << /0 0 /3 1 >> >> >> << /0 << /'
b'0 (\xfe\xff\x00G\x00y\x00o\x00m\x00a\x00t\x00s\x00u\x00Y\x00a\x00k\x00u\x00m'
b'\x00o\x00n\x00o\x00H\x00a\x00n\x00k\x00a\x00k\x00u) /5 << /0 0 /3 3 >> >> >> '
b'<< /0 << /0 (\xfe\xff\x00G\x00y\x00o\x00m\x00a\x00t\x00s\x00u\x00Y\x00a\x00k'
b'\x00u\x00m\x00o\x00n\x00o\x00Z\x00e\x00n\x00k\x00a\x00k\x00u) /5 << /0 0 /3 4'
b' >> >> >> << /0 << /0 (\xfe\xff\x00Y\x00a\x00k\x00u\x00m\x00o\x00n\x00o\x00Z'
b'\x00e\x00n\x00k\x00a\x00k\x00u) /5 << /0 0 /3 2 >> >> >> ] /1 [ << /0 0 >> <<'
b' /0 1 >> << /0 2 >> << /0 3 >> << /0 4 >> << /0 5 >> << /0 6 >> << /0 7 >> ] '
b'>> /4 << /0 [ << /0 << /0 (\xfe\xff\x00N\x00o\x00n\x00e) /5 << /0 (\xfe\xff) '
b'/1 (\xfe\xff) /2 (\xfe\xff) /3 (\xfe\xff) /4 0 >> >> >> << /0 << /0 (\xfe\xff'
b'\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00s\x00o\x00k'
b'\x00u\x00H\x00a\x00r\x00d) /5 << /0 (\xfe\xff\x00!\x00\\)\x00,\x00.\x00:\x00;'
b'\x00?\x00]\x00}\x00\xa2 \x14 \x19 \x1d 0!\x03!\t0\x010\x020\x050\t0\x0b0\n0'
b'\x0f0\x110\x150A0C0E0G0I0c0\x830\x850\x870\x8e0\x9b0\x9c0\x9d0\x9e0\xa10\xa30'
b'\xa50\xa70\xa90\xc30\xe30\xe50\xe70\xee0\xf50\xf60\xfb0\xfc0\xfd0\xfe\xff\x01'
b'\xff\x05\xff\t\xff\x0c\xff\x0e\xff\x1a\xff\x1b\xff\x1f\xff=\xff]) /1 (\xfe'
b'\xff\x00\\(\x00[\x00{\x00\xa3\x00\xa7 \x18 \x1c0\x080\n0\x0c0\x0e0\x100\x120'
b'\x14\xff\x03\xff\x04\xff\x08\xff \xff;\xff[\xff\xe5) /2 (\xfe\xff \x14 % &) /'
b'3 (\xfe\xff0\x010\x02\xff\x0c\xff\x0e) /4 1 >> >> >> << /0 << /0 (\xfe\xff'
b'\x00P\x00h\x00o\x00t\x00o\x00s\x00h\x00o\x00p\x00K\x00i\x00n\x00s\x00o\x00k'
b'\x00u\x00S\x00o\x00f\x00t) /5 << /0 (\xfe\xff \x19 \x1d0\x010\x020\x050\t0'
b'\x0b0\n0\x0f0\x110\x150\x9d0\x9e0\xfb0\xfd0\xfe\xff\x01\xff\t\xff\x0c\xff\x0e'
b'\xff\x1a\xff\x1b\xff\x1f\xff=\xff]) /1 (\xfe\xff \x18 \x1c0\x080\n0\x0c0\x0e0'
b'\x100\x14\xff\x08\xff;\xff[) /2 (\xfe\xff \x14 % &) /3 (\xfe\xff0\x010\x02'
b'\xff\x0c\xff\x0e) /4 2 >> >> >> << /0 << /0 (\xfe\xff\x00H\x00a\x00r\x00d) /5'
b' << /0 (\xfe\xff\x00!\x00\\)\x00,\x00.\x00:\x00;\x00?\x00]\x00}\x00\xa2 \x14 '
b'\x19 \x1d 0!\x03!\t0\x010\x020\x050\t0\x0b0\n0\x0f0\x110\x150A0C0E0G0I0c0\x83'
b'0\x850\x870\x8e0\x9b0\x9c0\x9d0\x9e0\xa10\xa30\xa50\xa70\xa90\xc30\xe30\xe50'
b'\xe70\xee0\xf50\xf60\xfb0\xfc0\xfd0\xfe\xff\x01\xff\x05\xff\t\xff\x0c\xff\x0e'
b'\xff\x1a\xff\x1b\xff\x1f\xff=\xff]) /1 (\xfe\xff\x00\\(\x00[\x00{\x00\xa3\x00'
b'\xa7 \x18 \x1c0\x080\n0\x0c0\x0e0\x100\x120\x14\xff\x03\xff\x04\xff\x08\xff '
b'\xff;\xff[\xff\xe5) /2 (\xfe\xff \x14 % &) /3 (\xfe\xff0\x010\x02\xff\x0c\xff'
b'\x0e) /4 1 >> >> >> << /0 << /0 (\xfe\xff\x00S\x00o\x00f\x00t) /5 << /0 (\xfe'
b'\xff \x19 \x1d0\x010\x020\x050\t0\x0b0\n0\x0f0\x110\x150\x9d0\x9e0\xfb0\xfd0'
b'\xfe\xff\x01\xff\t\xff\x0c\xff\x0e\xff\x1a\xff\x1b\xff\x1f\xff=\xff]) /1 ('
b'\xfe\xff \x18 \x1c0\x080\n0\x0c0\x0e0\x100\x14\xff\x08\xff;\xff[) /2 (\xfe'
b'\xff \x14 % &) /3 (\xfe\xff0\x010\x02\xff\x0c\xff\x0e) /4 2 >> >> >> ] /1 [ <'
b'< /0 0 >> << /0 1 >> << /0 2 >> << /0 3 >> << /0 4 >> ] >> /5 << /0 [ << /0 <'
b'< /0 (\xfe\xff\x00N\x00o\x00r\x00m\x00a\x00l\x00 \x00R\x00G\x00B) /6 << /0 2 '
b'/1 12.0 /2 false /3 false /4 true /5 0.0 /6 1.0 /7 1.0 /8 0 /9 0.0 /10 0.0 /1'
b'1 1 /12 0 /13 0 /14 0 /15 0 /16 0 /17 0.0 /18 true /19 false /20 false /21 fa'
b'lse /22 false /23 false /24 false /25 false /26 false /27 false /28 false /29'
b' false /30 0 /31 false /32 false /33 false /34 false /35 2 /36 0.0 /37 2 /38 '
b'0 /39 0 /40 false /41 2 /42 0 /43 << /0 .5 >> /44 2 /45 2 /46 7 /47 0 /48 0 /'
b'49 -1.0 /50 -1.0 /51 0 /52 false /53 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 '
b'0.0 0.0 0.0 ] >> >> /54 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 0.0 0.0 0.0 ]'
b' >> >> /55 << /99 /SimpleBlender >> /56 true /57 false /58 true /59 false /60'
b' false /61 0 /62 0 /63 1.0 /64 4.0 /65 0.0 /66 [ ] /67 [ ] /68 0 /69 0 /70 0 '
b'/71 4 /72 0.0 /73 0.0 /74 false /75 false /76 false /77 true /78 true /79 << '
b'/99 /SimplePaint /0 << /0 1 /1 [ 1.0 1.0 1.0 0.0 ] >> >> /80 false /81 0 /82 '
b'3.0 /83 3.0 /84 false /85 0 /86 << /99 /SimpleCustomFeature >> /87 100.0 /88 '
b'true >> >> >> << /0 << /0 (\xfe\xff\x00N\x00o\x00n\x00e) /5 0 /6 << >> >> >> '
b'] /1 [ << /0 0 >> << /0 1 >> ] >> /6 << /0 [ << /0 << /0 (\xfe\xff\x00N\x00o'
b'\x00r\x00m\x00a\x00l\x00 \x00R\x00G\x00B) /5 << /0 0 /1 0.0 /2 0.0 /3 0.0 /4 '
b'0.0 /5 0.0 /6 1 /7 1.2 /8 0 /9 true /10 6 /11 2 /12 2 /13 0 /14 36.0 /15 true'
b' /16 .5 /17 [ .8 1.0 1.33 ] /18 [ 0.0 0.0 0.0 ] /19 [ 1.0 1.0 1.0 ] /20 6 /21'
b' false /22 0 /23 true /24 0 /25 0 /27 /nil /26 false /28 /nil /29 false /30 <'
b'< >> /31 36.0 /32 << >> /33 0 /34 7 /35 0 /36 /nil /37 0 /38 false /39 0 /40 '
b'2 >> >> >> << /0 << /0 (\xfe\xff\x00B\x00a\x00s\x00i\x00c\x00 \x00P\x00a\x00r'
b'\x00a\x00g\x00r\x00a\x00p\x00h) /5 << /0 0 /1 0.0 /2 0.0 /3 0.0 /4 0.0 /5 0.0'
b' /6 1 /7 1.2 /8 0 /9 true /10 6 /11 2 /12 2 /13 0 /14 36.0 /15 true /16 .5 /1'
b'7 [ .8 1.0 1.33 ] /18 [ 0.0 0.0 0.0 ] /19 [ 1.0 1.0 1.0 ] /20 6 /21 false /22'
b' 0 /23 true /24 0 /25 0 /27 /nil /26 false /28 /nil /29 false /30 << >> /31 3'
b'6.0 /32 << /0 2 /1 12.0 /2 false /3 false /4 true /5 0.0 /6 1.0 /7 1.0 /8 0 /'
b'9 0.0 /10 0.0 /11 1 /12 0 /13 0 /15 0 /16 0 /18 true /19 false /20 true /21 f'
b'alse /22 false /23 false /24 false /25 false /26 false /27 false /28 false /2'
b'9 false /30 0 /35 1 /38 0 /53 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 0.0 0.0'
b' 0.0 ] >> >> /54 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 0.0 0.0 0.0 ] >> >> '
b'/68 0 /70 1 /71 4 /72 0.0 /73 0.0 /87 0.0 >> /33 0 /34 7 /35 0 /36 /nil /37 0'
b' /38 false /39 0 /40 2 >> /6 0 >> >> ] /1 [ << /0 0 >> << /0 1 >> ] >> /8 << '
b'/0 [ << /0 << /2 << /6 [ -1.0 -1.0 ] /11 << /4 -1 /7 false >> >> >> >> ] >> /'
b'9 << /0 [ << /0 << /0 (\xfe\xff\x00k\x00P\x00r\x00e\x00d\x00e\x00f\x00i\x00n'
b'\x00e\x00d\x00N\x00u\x00m\x00e\x00r\x00i\x00c\x00L\x00i\x00s\x00t\x00S\x00t'
b'\x00y\x00l\x00e\x00T\x00a\x00g) /6 1 >> >> << /0 << /0 (\xfe\xff\x00k\x00P'
b'\x00r\x00e\x00d\x00e\x00f\x00i\x00n\x00e\x00d\x00U\x00p\x00p\x00e\x00r\x00c'
b'\x00a\x00s\x00e\x00A\x00l\x00p\x00h\x00a\x00L\x00i\x00s\x00t\x00S\x00t\x00y'
b'\x00l\x00e\x00T\x00a\x00g) /6 2 >> >> << /0 << /0 (\xfe\xff\x00k\x00P\x00r'
b'\x00e\x00d\x00e\x00f\x00i\x00n\x00e\x00d\x00L\x00o\x00w\x00e\x00r\x00c\x00a'
b'\x00s\x00e\x00A\x00l\x00p\x00h\x00a\x00L\x00i\x00s\x00t\x00S\x00t\x00y\x00l'
b'\x00e\x00T\x00a\x00g) /6 3 >> >> << /0 << /0 (\xfe\xff\x00k\x00P\x00r\x00e'
b'\x00d\x00e\x00f\x00i\x00n\x00e\x00d\x00U\x00p\x00p\x00e\x00r\x00c\x00a\x00s'
b'\x00e\x00R\x00o\x00m\x00a\x00n\x00N\x00u\x00m\x00L\x00i\x00s\x00t\x00S\x00t'
b'\x00y\x00l\x00e\x00T\x00a\x00g) /6 4 >> >> << /0 << /0 (\xfe\xff\x00k\x00P'
b'\x00r\x00e\x00d\x00e\x00f\x00i\x00n\x00e\x00d\x00L\x00o\x00w\x00e\x00r\x00c'
b'\x00a\x00s\x00e\x00R\x00o\x00m\x00a\x00n\x00N\x00u\x00m\x00L\x00i\x00s\x00t'
b'\x00S\x00t\x00y\x00l\x00e\x00T\x00a\x00g) /6 5 >> >> << /0 << /0 (\xfe\xff'
b'\x00k\x00P\x00r\x00e\x00d\x00e\x00f\x00i\x00n\x00e\x00d\x00B\x00u\x00l\x00l'
b'\x00e\x00t\x00L\x00i\x00s\x00t\x00S\x00t\x00y\x00l\x00e\x00T\x00a\x00g) /6 6 '
b'>> >> ] /1 [ << /0 0 >> << /0 1 >> << /0 2 >> << /0 3 >> << /0 4 >> << /0 5 >'
b'> ] >> >> /1 << /0 << /0 << /0 1 /1 [ << /0 (\xfe\xff\x00 ) /1 (\xfe\xff\x001'
b') >> << /0 (\xfe\xff\x00\r) /1 (\xfe\xff\x006) >> << /0 (\xfe\xff\x00\t) /1 ('
b'\xfe\xff\x000) >> << /0 (\xfe\xff \\)) /1 (\xfe\xff\x005) >> << /0 (\xfe\xff'
b'\x00\x03) /1 (\xfe\xff\x005) >> << /0 (\xfe\xff0\x00) /1 (\xfe\xff\x001) >> <'
b'< /0 (\xfe\xff\x00\xad) /1 (\xfe\xff\x003) >> ] >> /1 0 /2 0 /3 .583 /4 .333 '
b'/5 .583 /6 .333 /7 .7 /8 true /9 [ << /0 0 /1 (\xfe\xff \x1c) /2 (\xfe\xff '
b'\x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 1 /1 (\xfe\xff \x1d) /2 '
b'(\xfe\xff \x1d) /3 (\xfe\xff \x19) /4 (\xfe\xff \x19) >> << /0 2 /1 (\xfe\xff'
b'\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /'
b'0 3 /1 (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe'
b'\xff \x19) >> << /0 4 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a'
b') /4 (\xfe\xff \x18) >> << /0 5 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 ('
b'\xfe\xff \x1a) /4 (\xfe\xff \x18) >> << /0 6 /1 (\xfe\xff\x00\xab) /2 (\xfe'
b'\xff\x00\xbb) /3 (\xfe\xff 9) /4 (\xfe\xff :) >> << /0 7 /1 (\xfe\xff \x1c) /'
b'2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 8 /1 (\xfe'
b'\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> '
b'<< /0 9 /1 (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 ('
b'\xfe\xff \x19) >> << /0 10 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff'
b' \x18) /4 (\xfe\xff \x19) >> << /0 11 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /'
b'3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 12 /1 (\xfe\xff \x1c) /2 (\xfe'
b'\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 13 /1 (\xfe\xff '
b'\x1d) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x19) /4 (\xfe\xff \x19) >> << /0 14 /1'
b' (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> '
b'<< /0 15 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe'
b'\xff \x19) >> << /0 16 /1 (\xfe\xff \x1d) /2 (\xfe\xff \x1d) /3 (\xfe\xff '
b'\x19) /4 (\xfe\xff \x19) >> << /0 17 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3'
b' (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 18 /1 (\xfe\xff\x00\xab) /2 ('
b'\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 19 /1 (\xfe'
b'\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> '
b'<< /0 20 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a) /4 (\xfe'
b'\xff \x18) >> << /0 21 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff '
b'\x1a) /4 (\xfe\xff \x18) >> << /0 22 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3'
b' (\xfe\xff \x1a) /4 (\xfe\xff \x18) >> << /0 23 /1 (\xfe\xff \x1e) /2 (\xfe'
b'\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 24 /1 (\xfe\xff '
b'\x1e) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x1a) /4 (\xfe\xff \x19) >> << /0 25 /1'
b' (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff 9) /4 (\xfe\xff :) >> '
b'<< /0 26 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe'
b'\xff \x19) >> << /0 27 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff '
b'\x18) /4 (\xfe\xff \x19) >> << /0 28 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1d) /3'
b' (\xfe\xff \x19) /4 (\xfe\xff \x19) >> << /0 29 /1 (\xfe\xff0\x1d) /2 (\xfe'
b'\xff0\x1e) >> << /0 30 /1 (\xfe\xff0\x0c) /2 (\xfe\xff0\r) >> << /0 31 /1 ('
b'\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> <<'
b' /0 32 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff '
b'\x19) >> << /0 33 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a) /4'
b' (\xfe\xff \x18) >> << /0 34 /1 (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 ('
b'\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 35 /1 (\xfe\xff \x1c) /2 (\xfe\xff'
b' \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> << /0 36 /1 (\xfe\xff \x1e) /'
b'2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a) /4 (\xfe\xff \x18) >> << /0 37 /1 (\xfe'
b'\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff \x18) /4 (\xfe\xff \x19) >> '
b'<< /0 38 /1 (\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a) /4 (\xfe'
b'\xff \x18) >> << /0 39 /1 (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe'
b'\xff\x00<) /4 (\xfe\xff\x00>) >> << /0 40 /1 (\xfe\xff \x1e) /2 (\xfe\xff '
b'\x1c) /3 (\xfe\xff \x1a) /4 (\xfe\xff \x18) >> << /0 41 /1 (\xfe\xff\x00\xab)'
b' /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff\x00<) /4 (\xfe\xff\x00>) >> << /0 42 /1 ('
b'\xfe\xff \x1e) /2 (\xfe\xff \x1c) /3 (\xfe\xff \x1a) /4 (\xfe\xff \x18) >> <<'
b' /0 43 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe\xff \x18) /4 (\xfe\xff '
b'\x19) >> << /0 44 /1 (\xfe\xff\x00\xab) /2 (\xfe\xff\x00\xbb) /3 (\xfe\xff 9)'
b' /4 (\xfe\xff :) >> << /0 45 /1 (\xfe\xff \x1c) /2 (\xfe\xff \x1d) /3 (\xfe'
b'\xff \x18) /4 (\xfe\xff \x19) >> ] /15 << /0 (\xfe\xff\x00H\x00u\x00n\x00s'
b'\x00p\x00e\x00l\x00l) >> /16 false >> /1 [ << /0 << /0 (\xfe\xff\x00t\x00e'
b'\x00s\x00t\x00 \x00s\x00t\x00r\x00i\x00n\x00g\x00\r\x00o\x00t\x00h\x00e\x00r'
b'\x00 \x00l\x00i\x00n\x00e\x00\r\x00t\x00h\x00i\x00r\x00d\x00 \x00l\x00i\x00n'
b'\x00e\x00\r) /5 << /0 [ << /0 << /0 << /0 (\xfe\xff) /5 << /0 2 /33 1 >> /6 1'
b' >> >> /1 12 >> << /0 << /0 << /0 (\xfe\xff) /5 << /0 2 /33 1 >> /6 1 >> >> /'
b'1 11 >> << /0 << /0 << /0 (\xfe\xff) /5 << /0 2 /33 1 >> /6 1 >> >> /1 11 >> '
b'] >> /6 << /0 [ << /0 << /0 << /0 (\xfe\xff) /5 1 /6 << /0 0 /1 48.0 /38 14 /'
b'53 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 .74547 .47001 .69903 ] >> >> >> >>'
b' >> /1 34 >> ] >> /7 0 /10 << /0 4 /2 true >> >> /1 << /0 [ << /0 0 >> ] /1 <'
b'< /0 [ << /0 << /1 1 >> /1 12 >> << /0 << /1 1 >> /1 11 >> << /0 << /1 1 >> /'
b'1 11 >> ] >> /2 [ << /99 /PC /5 0 /6 [ << /99 /F /10 0 /5 2 /6 [ << /99 /R /6'
b' [ << /99 /R /5 2 /6 [ << /99 /L /14 -41.23169 /15 14.71875 /1 [ 0.0 -41.2316'
b'9 0.0 14.71875 ] /5 3 /6 [ << /99 /S /15 << /0 12 /2 0 /5 false >> /0 << /0 ['
b' -108.45703 0.0 ] >> /6 [ << /99 /G /1 [ 0.0 -41.23169 216.91406 14.71875 ] /'
b'5 [ 87 72 86 87 3 86 87 85 76 81 74 3 ] /8 [ -108.45703 -41.23169 108.45703 1'
b'4.71875 ] /9 [ -108.45703 -41.23169 121.88672 14.71875 ] /10 << /0 [ << /4 2 '
b'/9 0 /10 0 >> << /0 1 /4 2 /9 0 /10 0 >> ] /1 [ 1 11 ] >> /11 true /12 -41.23'
b'169 /13 14.71875 /20 0 >> ] >> ] >> << /99 /L /10 57.6 /14 -41.23169 /15 14.7'
b'1875 /0 << /0 [ 0.0 57.6 ] >> /1 [ 0.0 -41.23169 0.0 14.71875 ] /5 3 /6 [ << '
b'/99 /S /15 << /0 11 /2 0 /5 false >> /0 << /0 [ -107.98828 0.0 ] >> /6 [ << /'
b'99 /G /1 [ 0.0 -41.23169 215.97656 14.71875 ] /5 [ 82 87 75 72 85 3 79 76 81 '
b'72 3 ] /8 [ -107.98828 16.36831 107.98828 72.31876 ] /9 [ -107.98828 16.36831'
b' 121.41797 72.31876 ] /11 true /12 -41.23169 /13 14.71875 /20 0 >> ] >> ] >> '
b'<< /99 /L /10 115.2 /14 -41.23169 /15 14.71876 /0 << /0 [ 0.0 115.2 ] >> /1 ['
b' 0.0 -41.23169 0.0 14.71875 ] /5 3 /6 [ << /99 /S /15 << /0 11 /2 0 /5 false '
b'>> /0 << /0 [ -98.85938 0.0 ] >> /6 [ << /99 /G /1 [ 0.0 -41.23169 197.71875 '
b'14.71875 ] /5 [ 87 75 76 85 71 3 79 76 81 72 3 ] /8 [ -98.85938 73.96832 98.8'
b'5938 129.91876 ] /9 [ -98.85938 73.96832 112.28906 129.91876 ] /11 true /12 -'
b'41.23169 /13 14.71875 /20 0 >> ] >> ] >> ] >> ] >> ] >> ] >> ] >> >> ] /2 << '
b'/0 2 /1 12.0 /2 false /3 false /4 true /5 0.0 /6 1.0 /7 1.0 /8 0 /9 0.0 /10 0'
b'.0 /11 1 /12 0 /13 0 /14 0 /15 0 /16 0 /17 0.0 /18 true /19 false /20 false /'
b'21 false /22 false /23 false /24 false /25 false /26 false /27 false /28 fals'
b'e /29 false /30 0 /31 false /32 false /33 false /34 false /35 2 /36 0.0 /37 2'
b' /38 0 /39 0 /40 false /41 2 /42 0 /43 << /0 .5 >> /44 2 /45 2 /46 7 /47 0 /4'
b'8 0 /49 -1.0 /50 -1.0 /51 0 /52 false /53 << /99 /SimplePaint /0 << /0 1 /1 ['
b' 1.0 0.0 0.0 0.0 ] >> >> /54 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 0.0 0.0 '
b'0.0 ] >> >> /55 << /99 /SimpleBlender >> /56 true /57 false /58 true /59 fals'
b'e /60 false /61 0 /62 0 /63 1.0 /64 4.0 /65 0.0 /66 [ ] /67 [ ] /68 0 /69 0 /'
b'70 0 /71 4 /72 0.0 /73 0.0 /74 false /75 false /76 false /77 true /78 true /7'
b'9 << /99 /SimplePaint /0 << /0 1 /1 [ 1.0 1.0 1.0 0.0 ] >> >> /80 false /81 0'
b' /82 3.0 /83 3.0 /84 false /85 0 /86 << /99 /SimpleCustomFeature >> /87 100.0'
b' /88 true >> /3 << /0 0 /1 0.0 /2 0.0 /3 0.0 /4 0.0 /5 0.0 /6 1 /7 1.2 /8 0 /'
b'9 true /10 6 /11 2 /12 2 /13 0 /14 36.0 /15 true /16 .5 /17 [ .8 1.0 1.33 ] /'
b'18 [ 0.0 0.0 0.0 ] /19 [ 1.0 1.0 1.0 ] /20 6 /21 false /22 0 /23 true /24 0 /'
b'25 0 /27 /nil /26 false /28 /nil /29 false /30 << >> /31 36.0 /32 << >> /33 0'
b' /34 7 /35 0 /36 /nil /37 0 /38 false /39 0 /40 2 >> >>\x008B64FMsk\x00\x00'
b'\x00\x00\x00\x00\x00\x0c\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x002'
)

# grouping (has_bold, has_italic, has_bold_italic) -> font names group for these styles
# generated by parse_adobe_fonts.py
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

# mapping font_name -> tuple (font_name_for_bold, font_name_for_italic, font_name_for_bold_italic). font names in the tuple can be None.
table_regular_font_to_styled = {}
def init_table_regular_font_to_styled():
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
            assert 0 <= i < 2**32, (i, "size is too large for 32-bit psd, try Big Psd output (PSB)")
        f.write(i.to_bytes(4*psd_version, 'big'))

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
        has_pixel_data = layer_type == "lt_bitmap" and None != layer_bitmaps.get(layer.MainId) and layer_bitmaps[layer.MainId].LayerBitmap
        if layer_type == "lt_bitmap" and not filter_layer_info and not has_fill_color and has_pixel_data:
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

    def add_fill_color_for_background_layer(layer_tags, layer):
        has_fill_color = getattr(layer, 'DrawColorEnable', None)
        if has_fill_color:
            fill_color = [ getattr(layer, 'DrawColorMain' + x, 0)/(2**32)*255 for x in ['Red', 'Green', 'Blue'] ]
            obj = PsdObjDescriptorWriter(io.BytesIO())
            write_int(obj.f, 16) # descriptor version
            obj.write_obj_header('null', 1)
            obj.write_field('Clr ', 'Objc')
            obj.write_obj_header('RGBC', 3)
            obj.write_doub('Rd  ', fill_color[0])
            obj.write_doub('Grn ', fill_color[1])
            obj.write_doub('Bl  ', fill_color[2])
            layer_tags.append((b'SoCo', obj.f.getvalue()))


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
                

    def export_layer(f, layer_entry, txt, make_invisible):
        layer_type, layer = layer_entry
        logging.info('exporting %s', layer.LayerName if layer else '-')

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

        add_filter_layer_info(layer_tags, layer)

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
        def write_obj_header(self, class_id, field_count, write4as0 = True):
            self.write_unicode_str('') # empty class name
            self.write_key(class_id, write4as0)
            write_int(self.f, field_count) # obj field count
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
        def write_untf(self, field_name, value):
            self.write_field(field_name, 'UntF')
            self.f.write(b'#Pnt')
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
            obj.write_untf('Left', left)
            obj.write_untf('Top ', top)
            obj.write_untf('Rght', right)
            obj.write_untf('Btom', bottom)

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

            if not (bool(text_info) and cmd_args.text_layer_raster == "disable"):
                make_invisible = bool(text_info) and cmd_args.text_layer_raster == "invisible"
                channels_data.append(export_layer(f, layer_entry, None, make_invisible))

            if cmd_args.text_layer_vector != 'disable':
                for txt in text_info:
                    _, layer = layer_entry
                    make_invisible = cmd_args.text_layer_vector == 'invisible'
                    channels_data.append(export_layer(f, ('lt_text', layer), txt, make_invisible))

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

    logging.info("PSD export done")

    # less garbge in output and command line - put temporary sqllite near result psd (keep if asked), don't write layer_info.txt, write pngs if asked from cmd-line

    # FilterLayerInfo - parse adjusments layers

    # layer color ( lclr )

    # optimize rle output - remove margins

    #5) write layer effects (stroke):
    #6) solid fill, background, gradients layers


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
    parser.add_argument('--keep-sqlite', help='Keep the temporary SQLite file, path is derived from output psd or output directory if not specified with --sqlite-file', action='store_true')
    parser.add_argument('--text-layer-raster', help='Export text layers as raster: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='enable')
    parser.add_argument('--text-layer-vector', help='Export text layers as vector: enable, disable, invisible.', choices=['enable', 'disable', 'invisible'], default='invisible')
    parser.add_argument('--ignore-zlib-errors', help='ignore decompression error for damaged data', action='store_true')
    parser.add_argument("--blank-psd-preview", help="--don't generate psd thumbnail preview (this allows to avoid Image module import, and works faster/gives smaller output file)", action='store_true')
    parser.add_argument('--psd-empty-bitmap-data', help='export bitmaps as empty to psd, usefull for faster export when pixel data is not needed', action='store_true')

    cmd_args = parser.parse_args()

    init_logging(cmd_args.log_level)

    logging.debug('command line: %s', sys.argv)

    if cmd_args.sqlite_file:
        cmd_args.keep_sqlite = True

    if not (cmd_args.keep_sqlite or cmd_args.output_dir or cmd_args.output_psd):
        parser.error('At least one output must be specified: --keep-sqlite, --output-dir, or --output.')

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
        else:
            parser.error('Output SQLite file path cannot be derived. Please specify --sqlite-file or an output option.')

    if cmd_args.output_dir:
        if not os.path.isdir(cmd_args.output_dir):
            os.mkdir(cmd_args.output_dir)


def main():
    parse_command_line()
    extract_csp(cmd_args.input_file)

if __name__ == "__main__":
    main()
