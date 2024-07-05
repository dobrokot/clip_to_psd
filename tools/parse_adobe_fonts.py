

# This scipt parses Adobe fonts file description which defines font styles for fonts used in PSD files.

# Information useful to search for such a file: this files could be at 
# c:/Users/<user>/AppData/Local/Adobe/TypeSupport/CS6/AdobeFnt_OSFonts.lst 
# and it contains lines such as "%BeginFont"

# The output has table which defines bold/italic/bold_italic styles for each font family from the file.
# See comments about font_name_regular_to_style_font_name table in clip_to_psd.py to get 
# exact output example and structure description.

import sys
import itertools

def get_normalized_style_fonts(family_fonts):
    # if font doesn't have standard regular/bold/italic/bold_italic styles, 
    # try to fill them from semibold/demibold/medium/book/italic_bold styles

    style2font = {}
    for f in family_fonts:
        style2font.setdefault(f['StyleName'].lower(), []).append(f['FontName'])

    for bold_variant in ['semibold', 'demibold']:
        if 'bold' not in style2font and bold_variant in style2font:
            style2font['bold'] = style2font[bold_variant]
            del style2font[bold_variant]
        if 'bold italic' not in style2font and (bold_variant + ' italic') in style2font:
            style2font['bold italic'] = style2font[bold_variant + ' italic']
            del style2font[bold_variant + ' italic']

    if 'regular' not in style2font:
        for regular_variant in ('medium', 'book'):
            if regular_variant in style2font:
                style2font['regular'] = style2font[regular_variant]
                del style2font[regular_variant]
                break

    if 'italic bold' in style2font and 'bold italic' not in style2font:
        style2font['bold italic'] = style2font['italic bold'] 
        del style2font['italic bold'] 

    if 'regular' not in style2font:
        print("warning: font style has no 'Regular' style or its equivalent" , file = sys.stderr)

    # keep fonts only special for style, don't duplicate regular in special styles (they need FauxBold/FauxItalic). 
    # Exampe of font with duplicated font name style: "Raavi"
    regular_fonts = style2font['regular']
    for style, fonts in list(style2font.items()):
        if style != 'regular':
            fonts = [f for f in fonts if f not in regular_fonts] 
            if fonts:
                style2font[style] = fonts
            else:
                del style2font[style]

    return style2font


# search for font_name_regular_to_style_font_name in clip_to_psd.py to get information about what this function returns and example of its output.
def add_font_style_info(style2font, font_name_regular_to_style_font_name):
    #pylint: disable=cell-var-from-loop
    def font_name(style):
        return style2font[style][0] 
    def has_font_for(style):
        return style in style2font and len(style2font[style]) >= 0

    #style2standard_name = dict()
    style_and_filename_parts_all = [('bold', '-Bold'), ('italic', '-Italic'), ('bold italic', '-BoldItalic')]

    font_styles_bitset = [has_font_for(x[0]) for x in style_and_filename_parts_all]
    #has_bold, has_italic, has_bold_italic = font_styles_bitset

    if not any(font_styles_bitset):
        return
    if 'regular' not in style2font:
        return

    assert len(style_and_filename_parts_all) == len(font_styles_bitset)
    style_and_filename_parts = [x for (x, flag) in zip(style_and_filename_parts_all, font_styles_bitset) if flag]
    

    for regular_font_name in style2font['regular']:
        def check_commmon_pattern(style_and_filename_parts, remove_dash):
            base_names = []
            for style, font_name_part in style_and_filename_parts:
                if remove_dash:
                    font_name_part = font_name_part.removeprefix('-')
                base_name = None
                if font_name_part in font_name(style) and font_name(style).replace(font_name_part, '') == regular_font_name:
                    assert '%' not in font_name(style)
                    base_name = font_name(style).replace(font_name_part, '%')
                base_names.append(base_name)
            if all(name and (name == base_names[0]) for name in base_names):
                return base_names[0]
            return None

        font_names_with_styles = font_name_regular_to_style_font_name.setdefault( tuple(font_styles_bitset), {} )
        if base_name := check_commmon_pattern(style_and_filename_parts, remove_dash = False):
            font_names_with_styles.setdefault('style_dash', []).append(base_name)
        elif base_name := check_commmon_pattern(style_and_filename_parts, remove_dash = True):
            font_names_with_styles.setdefault('style_no_dash', []).append(base_name)
        else:
            font_info = [font_name('regular')] + [font_name(style) for style, _ in style_and_filename_parts]
            font_names_with_styles.setdefault('style_tuple', []).append(font_info)

def fix_arial_narrow(fonts_list):
    # microsoft added 4 same styles "Narrow" to FontFamily=Arial, and this makes programs confuse with standard 'arial' font in same font family 
    # without possibility to select bold/italic style.
    for f in fonts_list:
        if f['FamilyName'] == 'Arial' and f['StyleName'] == 'Narrow' and f['FontName'].startswith('ArialNarrow'):
            if '-' in f['FontName']:
                name_parts = f['FontName'].split('-')
                assert len(name_parts) == 2 and name_parts[0] == 'ArialNarrow', (name_parts)
                style = name_parts[1]
                if style != 'Italic' and style.endswith('Italic'):
                    style = style.removesuffix('Italic') + ' Italic'
            else:
                assert f['FontName'] == 'ArialNarrow'
                style = 'Regular'
            f['StyleName'] = style
            f['FamilyName'] = 'ArialNarrow'


def print_result_font_style_info(font_name_regular_to_style_font_name):
    print('# grouping (has_bold, has_italic, has_bold_italic) -> font names group for these styles')

    print('{')
    for font_styles_bitset in itertools.product((True, False), repeat = 3):
        font_names_with_styles = font_name_regular_to_style_font_name.get(font_styles_bitset)
        if font_names_with_styles:
            print(font_styles_bitset, ':')
            print('    {')
            print('       "dash":', font_names_with_styles.get('style_dash', []), ',')
            print('       "no_dash":', font_names_with_styles.get('style_no_dash', []), ',')
            print('       "tuple": [', )
            for x in font_names_with_styles['style_tuple']:
                print(' '*11, tuple(x), ',')
            print('    ]},')
    print('}')


def parse_input_file():
    fields = ['FontName', 'FamilyName', 'StyleName', 'StyleBits', 'WeightClass', 'WinName', 'FullName']

    fonts_list = []
    is_font_section = False
    font_dict = {}

    file_name = sys.argv[1] # c:/Users/<user>/AppData/Local/Adobe/TypeSupport/CS6/AdobeFnt_OSFonts.lst

    def process_line(line, font_dict):
        parts = line.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            key = key.strip()
            value = value.strip()
            if key in fields:
                font_dict[key] = int(value) if key == 'StyleBits' else value

    with open(file_name, 'r', encoding='UTF-8') as file:
        for line in file:
            line = line.strip()
            if line == '%BeginFont':
                is_font_section = True
                font_dict = {}  # Initialize a new dictionary for the font section
            elif line == '%EndFont':
                is_font_section = False
                if all(field in font_dict for field in fields):
                    fonts_list.append(font_dict)
            elif is_font_section:
                process_line(line, font_dict)

    return fonts_list

def main():
    fonts_list = parse_input_file()

    fix_arial_narrow(fonts_list)

    family2style = {}
    for f in fonts_list:
        k = f['FamilyName']
        family2style.setdefault(k, []).append(f)

    font_name_regular_to_style_font_name = {}

    for _familyname, family_fonts in sorted(family2style.items()):
        if len(family_fonts) > 1:
            style2font = get_normalized_style_fonts(family_fonts)
            add_font_style_info(style2font, font_name_regular_to_style_font_name)

    print_result_font_style_info(font_name_regular_to_style_font_name)

main()
