import ast
import os
import re
from collections import Counter
from subprocess import Popen, PIPE
import docx2txt
import textract
from PIL import Image
from bs4 import BeautifulSoup as bs
from celery.exceptions import SoftTimeLimitExceeded
from pdf2image import convert_from_path
from tika import parser
from RP_parser.helper_files.constants import SECTIONS
from utils.logging_helpers import info_logger

os.environ["TIKA_CLIENT_ONLY"] = "True"
os.environ["TIKA_SERVER_ENDPOINT"] = str(os.getenv("TIKA_SERVER_ENDPOINT"))
os.environ["TIKA_VERSION"] = str(os.getenv("TIKA_VERSION"))

Image.MAX_IMAGE_PIXELS = 1000000000
TEXT_EXTRACTION_TYPE = ast.literal_eval(os.getenv("TEXT_EXTRACTION_TYPE"))


def create_paragraphs(file_data_content):
    """
    Based on textlines pattern it will create paragraph.
    Args:
        file_data_content: list of text line.
    """
    lines = file_data_content.splitlines(True)
    paragraph = []
    for line in lines:
        if line.isspace():
            if paragraph:
                yield "".join(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield "".join(paragraph)


def extract_text_from_txt(file_path):
    """
    Extracting text form txt file
    Args:
        file_path: local path of file

    Returns:
        extracted text from doc
    """
    try:
        text = ""
        with open(file_path) as file:
            text += file.read()
        return list(create_paragraphs(text))
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return []


def extract_text_from_pdf_textract(file_path):
    """
    Extracting text form pdf
    textract(poppler) used for getting text from pdf. if pdf is image based then it will use ocr for that
    Args:
        file_path: local path of file

    Returns:
        extracted text from pdf
    """
    try:
        data = textract.process(file_path, method="pdftotext").decode("utf-8")
        # parsed_pdf = parser.from_file(file_path)
        # data = parsed_pdf['content']
        if not data:
            result = extract_text_from_img_pdf(file_path)
            return result, TEXT_EXTRACTION_TYPE["NE&E_IMAGE_PDF"]
        if len(data) <= 100:
            info_logger(
                "Might be combination of pdf and image ", file_path
            )  # when it's combination of img and pdf formate
            result = extract_text_from_img_pdf(file_path)
            return result, TEXT_EXTRACTION_TYPE["NE&E_IMAGE_PDF"]
        return data, TEXT_EXTRACTION_TYPE["NE&E_TEXTRACT"]
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None, None


def parse_sections(texts):
    """
    find  is it perfect sections or not? based on distance on text using section title dictionary logic
    Args:
        texts: list of text lines of resume
    Returns:
        True or False
    """
    line_label = {}
    label_gap = []
    text_lines = [line for line in texts.split("\n") if line.strip() != ""]
    start = 0
    end = len(text_lines)
    flags = dict.fromkeys(list(SECTIONS.keys()) + ["other"], False)
    flags["other"] = True
    for line_no, line in enumerate(text_lines):
        line = re.sub(r"[^\w\s]", "", line.lower()).strip()
        for clas, class_objs in SECTIONS.items():
            if line in class_objs:
                for active_class, flag in flags.items():
                    if flag:
                        if active_class != clas:
                            if active_class in line_label.keys():
                                line_label[active_class].append(
                                    [start, line_no]
                                )
                                label_gap.append(line_no - start)
                            else:
                                line_label[active_class] = [[start, line_no]]
                                label_gap.append(line_no - start)
                            flags[active_class] = False
                        flags[clas] = True
                        start = line_no
    for active_class, flag in flags.items():
        if flag:
            if active_class in line_label.keys():
                line_label[active_class].append([start, end])
                label_gap.append(end - start)
            else:
                line_label[active_class] = [[start, end]]
                label_gap.append(end - start)
            flags[active_class] = False
    label_gap_counter = Counter(label_gap)
    if label_gap_counter[0] > 0 or label_gap_counter[1] > 0:  # more than 2
        return False
    else:
        return True


def extract_text_from_pdf(file_path):
    """
    Extracting text from pdf
    textract(poppler) used for getting text from pdf. if pdf is image based then it will use ocr for that and flag
    Args:
        file_path: local path of file

    Returns:
        text: extracted text from pdf
        flag: [0: tika, 1: textract, 2: C,3: NE&E Image PDF, 4:Textract (Tika Failed), 5:NE&E Textract
            None: for other categories and error]
    """
    try:
        parsed_pdf = parser.from_file(file_path)
        tika_data = parsed_pdf["content"]
        textract_data = textract.process(file_path, method="pdftotext").decode(
            "utf-8"
        )
        if textract_data and tika_data:
            # Both data textract_text and tika_text
            if len(tika_data) <= 100 and len(textract_data) <= 100:
                # Both have no sufficient data so using extract_text_from_img_pdf
                info_logger(
                    "Might be combination of pdf and image ", file_path
                )
                result = extract_text_from_img_pdf(file_path)
                return result, TEXT_EXTRACTION_TYPE["IMAGE_PDF"]
            elif len(tika_data) <= 100 and len(textract_data) > 100:
                # tika_text have no sufficient data but textract_text have so returning textract_text
                return (
                    textract_data,
                    TEXT_EXTRACTION_TYPE["TEXTRACT_(TIKA_FAILED)"],
                )
            else:
                if parse_sections(tika_data):
                    # Dictionary logic will decide which formate is perfect
                    return tika_data, TEXT_EXTRACTION_TYPE["TIKA"]
                else:
                    return textract_data, TEXT_EXTRACTION_TYPE["TEXTRACT"]
        elif tika_data:
            # textract_text is failed but tika_text is available.
            if len(tika_data) <= 100:
                # textract_text is failed but tika_text is available
                # but not sufficient so using extract_text_from_img_pdf.
                info_logger(
                    "Might be combination of pdf and image ", file_path
                )  # When it's combination of img and pdf formate
                result = extract_text_from_img_pdf(file_path)
                return result, TEXT_EXTRACTION_TYPE["IMAGE_PDF"]
            else:
                return tika_data, TEXT_EXTRACTION_TYPE["TIKA"]
        else:
            # tika_text is failed but textract_text is available.
            if len(textract_data) <= 100:
                # tika_text is failed but textract_text is available.
                # but not sufficient so using extract_text_from_img_pdf.
                info_logger(
                    "Might be combination of pdf and image ", file_path
                )  # When it's combination of img and pdf formate
                result = extract_text_from_img_pdf(file_path)
                return result, TEXT_EXTRACTION_TYPE["IMAGE_PDF"]
            else:
                return textract_data, TEXT_EXTRACTION_TYPE["TEXTRACT"]
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None, None


def extract_text_from_doc(file_path):
    """
    Extracting text form doc
    inside that there is antiword binary who is helping to get proper text in manner of design
    Args:
        file_path: local path of file

    Returns:
        extracted text from doc
    """
    try:
        text = textract.process(file_path).decode("utf-8")
        return text
    except Exception as e:
        info_logger("Antiword gave error: ", e, file_path)
        try:
            process = Popen(["catdoc", file_path], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if stdout:
                return stdout.decode("utf-8")
            else:
                raise Exception("catdoc not able to parse")
        except Exception as e:
            try:
                result = extract_text_from_rtf(file_path)
                if len(result) == 0 or "<!DOCTYPE" in " ".join(result):
                    result = extract_text_using_bs(
                        file_path, encoding="ISO-8859-1"
                    )
                    return result
                else:
                    return result
            except Exception as e:
                info_logger(
                    "Got error while extracting text from: ", e, file_path
                )
                result = extract_text_from_docx(file_path)
                return result


def extract_text_from_docx(file_path):
    """
    Extracting text form docx using docx2txt
    Args:
        file_path: local path of file

    Returns:
        extracted text from docx
    """
    try:
        temp = docx2txt.process(file_path)
        text = [line.replace("\t", " ") for line in temp.split("\n") if line]
        text = "\n".join(text)
        return text
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None


def extract_text_from_img(file_path):
    """
    extracting text form img
    Args:
        file_path: local path of file

    Returns:
        extracted text from image
    """
    try:
        text = textract.process(file_path).decode("utf-8")
        return text
    except SoftTimeLimitExceeded as e:
        info_logger("Time limit Exceeded while converting text", e, file_path)
        return "TimeLimitExceeded"
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None


def extract_text_from_img_pdf(file_path):
    """
    Convert image pdf to image and use ocr for getting text
    Args:
        file_path: local path of file

    Returns:
        extracted text from image pdf

    """
    try:
        pdf = file_path.split("/")[-1]
        OP_MAIN_FOLDER = os.getenv("op_main_folder")
        pdf_name = ".".join(pdf.split(".")[:-1])
        pdf_name = pdf_name.replace(" ", "_").replace("(", "").replace(")", "")
        OP_FOLDER = os.path.join(OP_MAIN_FOLDER, pdf_name)
        if not os.path.exists(OP_FOLDER):
            os.makedirs(OP_FOLDER)
        pages = convert_from_path(file_path, fmt="png", dpi=500)
        for i in range(len(pages)):
            page = pages[i]
            output_path = os.path.join(
                OP_FOLDER,
                "{}_{}.png".format(
                    "".join(pdf.split(".")[:-1])
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", ""),
                    i,
                ),
            )
            page.save(output_path, "png")

        # Get pdf image to pages
        file_list = sorted(os.listdir(OP_FOLDER))
        image_filepaths = [
            os.path.join(OP_FOLDER, i) for i in file_list if i.endswith(".png")
        ]
        result = ""
        for i in image_filepaths:
            text = textract.process(i).decode("utf-8")
            result += text
        return result
    except SoftTimeLimitExceeded as e:
        info_logger("Time limit Exceeded while converting text", e, file_path)
        return "TimeLimitExceeded"
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None


def extract_text_from_rtf(file_path):
    """
    extracting text form rtf file
    Args:
        file_path: local path of file

    Returns:
        extracted text from image pdf

    """
    try:
        text = textract.process(file_path).decode("utf-8")
        return text
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None


def extract_text_using_bs(file_path, encoding=None):
    try:
        soup = bs(open(file_path, encoding=encoding).read())
        [s.extract() for s in soup(["style", "script"])]
        text = soup.get_text()
        return text
    except SoftTimeLimitExceeded as e:
        info_logger("Time limit Exceeded while converting text", e, file_path)
        return "TimeLimitExceeded"
    except Exception as e:
        info_logger("Got error while extracting text from: ", e, file_path)
        return None


def cleaning_text(text):
    """
    cleaning Text from unicode
    Args:
        text: list of uncleaned text
    Returns: list of cleaned text
    """
    result = []
    for line in text.split("\n"):
        line2 = line.strip()
        if line2 != "":
            result.append(line2)
    result = [
        re.sub(
            r"(?:^|(?<=(?<!\S)\S)) (?=\S(?!\S)|$)",
            "",
            line.encode("ascii", "ignore").decode(),
            flags=re.MULTILINE,
        )
        for line in result
    ]
    result = [" ".join(line.split()) for line in result]
    return result


def single_resume_read(file_path):
    """
    For converting any type of resume document to text
    Args:
        file_path: local path of file

    Returns:
        extracted text from resume doc

    """
    if os.path.isfile(file_path):
        try:
            txt = None
            flag = None
            if file_path.lower().endswith(".pdf"):
                txt, flag = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith(".docx"):
                txt = extract_text_from_docx(file_path)
            elif file_path.lower().endswith(".doc"):
                txt = extract_text_from_doc(file_path)
            elif file_path.lower().endswith(".txt"):
                txt = extract_text_from_txt(file_path)
            elif file_path.lower().endswith(".rtf"):
                txt = extract_text_from_rtf(file_path)
            else:
                if os.path.splitext(file_path)[1] in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".tiff",
                    ".tif",
                ]:
                    txt = extract_text_from_img(file_path)
            if txt:
                return cleaning_text(txt), flag
            else:
                return None, flag
        except Exception as e:
            info_logger("Got error while extracting text from: ", e, file_path)
            return None, None
    else:
        info_logger("File is not available at: ", file_path)
        return None, None
