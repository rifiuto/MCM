(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "array"
    "tabularx"
    "lipsum"
    "xparse")
   (LaTeX-add-labels
    "sec:test"))
 :latex)

