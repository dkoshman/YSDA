from constants import DATA_DIR, LATEX_PATH, PDFLATEX, GHOSTSCRIPT

import json
from multiprocessing import Pool
import os
import shutil
import string
import subprocess
import random
import tqdm


def generate_equation(latex, size, max_depth):
    """
    Generates a random latex equation
    -------
    params:
    :latex: -- dict with tokens to generate equation from
    :size: -- approximate size of equation
    :max_depth: -- max brackets and scope depth
    """

    tokens, pairs, scopes = latex["tokens"], latex["pairs"], latex["scope_manipulators"]

    def _generate_equation_recursive(size_left=size, depth_used=0):
        if size_left <= 0:
            return ""

        equation = ""
        (group,) = random.choices(
            [tokens, pairs, scopes],
            weights=[max_depth + 1, max_depth > depth_used, max_depth > depth_used],
        )

        if group is tokens:
            equation += " ".join(
                [
                    random.choice(tokens),
                    _generate_equation_recursive(size_left - 1, depth_used),
                ]
            )
            return equation

        post_scope_size = round(abs(random.gauss(0, size_left / 2)))
        size_left -= post_scope_size + 1

        if group is pairs:
            pair = random.choice(pairs)
            equation += " ".join(
                [
                    pair[0],
                    _generate_equation_recursive(size_left, depth_used + 1),
                    pair[1],
                    _generate_equation_recursive(post_scope_size, depth_used),
                ]
            )
            return equation

        elif group is scopes:
            scope_type, scope_group = random.choice(list(scopes.items()))
            scope_operator = random.choice(scope_group)
            equation += scope_operator

            if scope_type == "single":
                equation += "{ " + _generate_equation_recursive(
                    size_left, depth_used + 1
                )

            elif scope_type == "double_no_delimiters":
                equation += (
                    "{ "
                    + _generate_equation_recursive(size_left // 2, depth_used + 1)
                    + " } { "
                    + _generate_equation_recursive(size_left // 2, depth_used + 1)
                )

            elif scope_type == "double_with_delimiters":
                equation += (
                    "^ { "
                    + _generate_equation_recursive(size_left // 2, depth_used + 1)
                    + " } _ { "
                    + _generate_equation_recursive(size_left // 2, depth_used + 1)
                )

            equation += _generate_equation_recursive(post_scope_size, depth_used) + " }"

        return equation

    return _generate_equation_recursive()


def generate_image(
    directory, latex, filename, max_depth, equation_length, distribution_fraction
):
    """
    Generates a random tex file and corresponding image
    -------
    params:
    :directory: -- dir where to save files
    :latex: -- dict with parameters to generate tex
    :filename: -- absolute filename for the generated files
    :max_depth: -- max nested level of tex scopes
    :equation_length: -- max length of equation
    :distribution_fraction: -- fraction of whole available tex tokens to use
    """
    fracture = lambda sequence: sequence[
        : max(1, int(len(sequence) * distribution_fraction))
    ]
    for group in ["tokens", "pairs", "fonts", "font_sizes"]:
        latex[group] = fracture(latex[group])
    for key, value in list(latex["scope_manipulators"].items()):
        latex["scope_manipulators"]["key"] = fracture(value)

    size = random.randint((equation_length + 1) // 2, equation_length)
    equation = generate_equation(latex, size=size, max_depth=max_depth)

    font, font_options = random.choice(latex["fonts"])
    font_option = random.choice([""] + font_options)
    font_size = random.choice(latex["font_sizes"])
    template = string.Template(latex["template"])
    tex = template.substitute(
        font=font, font_option=font_option, fontsize=font_size, equation=equation
    )

    filepath = os.path.join(directory, filename)
    with open(f"{filepath}.tex", mode="w") as file:
        file.write(tex)

    try:
        pdflatex_process = subprocess.run(
            f"{PDFLATEX} -output-directory={directory} {filepath}.tex".split(),
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=1,
        )
    except subprocess.TimeoutExpired:
        os.remove(filepath + ".tex")
        return

    if pdflatex_process.returncode != 0:
        os.remove(filepath + ".tex")
        return

    subprocess.run(
        f"{GHOSTSCRIPT} -sDEVICE=png16m -dTextAlphaBits=4 -r200 -dSAFER -dBATCH -dNOPAUSE"
        f" -o {filepath}.png {filepath}.pdf".split(),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


def _generate_image_wrapper(args):
    return generate_image(*args)


def generate_data(
    examples_count, max_depth, equation_length, distribution_fraction
) -> None:
    """
    Clears a directory and generates a latex dataset in given directory
    """

    directory = os.path.abspath(DATA_DIR)
    shutil.rmtree(DATA_DIR)
    os.mkdir(DATA_DIR)

    with open(LATEX_PATH) as file:
        latex = json.load(file)

    filenames = set(
        f"{i:0{len(str(examples_count - 1))}d}" for i in range(examples_count)
    )
    files_before = set(os.listdir())

    while filenames:
        with Pool() as pool:
            list(
                tqdm.tqdm(
                    pool.imap(
                        _generate_image_wrapper,
                        (
                            (
                                directory,
                                latex,
                                filename,
                                max_depth,
                                equation_length,
                                distribution_fraction,
                            )
                            for filename in sorted(filenames)
                        ),
                    ),
                    "Generating images",
                    total=len(filenames),
                )
            )
        filenames -= set(
            os.path.splitext(filename)[0]
            for filename in os.listdir(directory)
            if filename.endswith(".png")
        )

    for file in (
        set(i.path for i in os.scandir(DATA_DIR)) | set(os.listdir()) - files_before
    ):
        if any(file.endswith(ext) for ext in [".aux", ".pdf", ".log", ".sh"]):
            os.remove(file)
