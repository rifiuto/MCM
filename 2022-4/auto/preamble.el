(TeX-add-style-hook
 "preamble"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("appendix" "toc" "page" "title" "titletoc" "header")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "appendix"
    "graphicx"
    "amsmath"
    "amssymb"
    "amsfonts"
    "amsthm"
    "xcolor"
    "geometry"
    "fancyhdr"
    "lastpage"
    "minted"
    "hyperref"
    "mathptmx"
    "lipsum")
   (TeX-add-symbols
    '("keywords" 1)
    "problem"
    "control"
    "team"
    "headset")
   (LaTeX-add-amsthm-newtheorems
    "Theorem"
    "Lemma"
    "Corollary"
    "Proposition"
    "Definition"
    "Example"))
 :latex)

