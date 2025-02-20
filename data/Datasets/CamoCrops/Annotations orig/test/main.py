import os
import xml.etree.ElementTree as ET


def replace_keyword_in_xml(file_path, old_keyword, new_keyword):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Function to recursively replace text in elements
    def replace_text(element):
        if element.text:
            element.text = element.text.replace(old_keyword, new_keyword)
        for child in element:
            replace_text(child)
            if child.tail:
                child.tail = child.tail.replace(old_keyword, new_keyword)

    # Replace the keyword in the root and its children
    replace_text(root)

    # Save the modified XML back to the same file
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


files = os.listdir()

for file in files:
    if file != 'main.py':
        replace_keyword_in_xml(file, 'green chili', 'green_chili')
        replace_keyword_in_xml(file, 'bell pepper', 'bell_pepper')
        replace_keyword_in_xml(file, 'french_beand', 'french_bean')
        replace_keyword_in_xml(file, 'french_beans', 'french_bean')
