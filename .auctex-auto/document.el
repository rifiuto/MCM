(TeX-add-style-hook
 "document"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("elsarticle" "review")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("caption" "font=small" "labelfont=bf" "labelsep=none") ("natbib" "numbers" "sort&compress") ("algorithm2e" "linesnumbered" "ruled" "vlined")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "elsarticle"
    "elsarticle10"
    "lineno"
    "hyperref"
    "amssymb"
    "graphicx"
    "caption"
    "wrapfig"
    "algorithm"
    "algpseudocode"
    "amsmath"
    "multirow"
    "float"
    "array"
    "longtable"
    "booktabs"
    "color"
    "subfigure"
    "tabu"
    "fancyhdr"
    "epstopdf"
    "ulem"
    "bm"
    "amsopn"
    "times"
    "geometry"
    "cite"
    "natbib"
    "cleveref"
    "setspace"
    "algorithm2e")
   (TeX-add-symbols
    '("pder" ["argument"] 1)
    '("tabincell" 2)
    "R"
    "llambda")
   (LaTeX-add-environments
    '("function" LaTeX-env-args ["argument"] 0)
    '("procedure" LaTeX-env-args ["argument"] 0)
    "myDef"
    "myTheo"
    "myRem"
    "myPro"))
 :latex)

