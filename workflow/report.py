import os
import argparse

def similarity_from_lumping(lumping):
    elements = lumping.split("_")
    if len(elements) == 2:
        if elements[0] == "t":
            d = "d^T_{ij}"
        elif elements[0] == "kl":
            d = "d^K_{ij}"
        if elements[1] == "q":
            g = "g^q_{ij}"
        elif elements[1] == "js":
            g = "g^J_{ij}"
        return f"{d} \\cdot {g}"
    else:
        if elements[0] == "t":
            s = "d^T_{ij}"
        elif elements[0] == "kl":
            s = "d^K_{ij}"
        elif elements[0] == "q":
            s = "g^q_{ij}"
        elif elements[0] == "js":
            s = "g^J_{ij}"
        return s

def report(system, lumping, rmsd=False):
    out = os.path.join("results", system, lumping)
    system = system.replace("_", r"\_")
    similarity = similarity_from_lumping(lumping)
    if rmsd:
        r = "\\includegraphics[width=0.53\\textwidth]{{\\rootdir rmsd}}"
        file_name = "report_rmsd.tex"
    else:
        r = ""
        file_name = "report.tex"
    tex = f"""
\\renewcommand{{\\rootdir}}{{{os.path.abspath(out)}/}}

\\begin{{figure}}
	\\centering
	\\includegraphics[width=0.3\\textwidth]{{\\rootdir sankey}}
	\\includegraphics[width=0.68\\textwidth]{{\\rootdir dendrogram}}
	\\includegraphics[width=0.6\\textwidth]{{\\rootdir ck_test}}
	\\includegraphics[width=0.35\\textwidth]{{\\rootdir timescales}}
	\\includegraphics[width=0.36\\textwidth]{{\\rootdir contacts}}
    {r}
	\\caption*{{{system}: \\similarity{{$S_{{ij}} = {similarity}$}}}}
\\end{{figure}}
\\begin{{landscape}}
	\\begin{{figure}}
		\\centering
		\\includegraphics[width=1.4\\textwidth]{{\\rootdir macrotraj}}
	    \\caption*{{{system}: \\similarity{{$S_{{ij}} = {similarity}$}}}}
	\\end{{figure}}
\\end{{landscape}}
    """

    with open(os.path.join(out, file_name), "w") as f:
        f.write(tex)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Perform MPP on MD simulation data",
        description=(
            "This program allows for the analysis of MD data utilizing the "
            "most probable path algorithm. It allows for easy plotting of "
            "different quality measures."
        ),
    )
    parser.add_argument(
        "system",
        help=(
            "Model name"
        )
    )
    parser.add_argument(
        "lumping",
        help=(
            "Lumping directory name (e.g. t_js)"
        )
    )
    parser.add_argument(
        "-r",
        "--rmsd",
        help=(
            "Add RMSD plot"
        ),
        action="store_true",
    )
    return parser.parse_args()
    

def main():
    args = parse_args()

    report(args.system, args.lumping, rmsd=args.rmsd)


if __name__ == "__main__":
    main()
