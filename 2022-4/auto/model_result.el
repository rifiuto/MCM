(TeX-add-style-hook
 "model_result"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("subfiles" "../mcmpaper")))
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
    "latex2e"
    "subfiles"
    "subfiles10")
   (LaTeX-add-labels
    "tab:4.1.1"
    "fig:4.2.1"
    "fig:4.2.4"
    "fig:4.2.5"
    "tab:4.3.1"
    "tab:4.3.2"
    "fig:4.4.1"
    "fig:4.4.a"
    "fig:5.1.1"
    "fig:5.1.2"
    "fig:5.1.a"
    "tab:5.1.1"
    "fig:5.2.1"
    "tab:5.2.1"
    "tab:5.2.2"
    "tab:6.2.1"
    "fig:6.2.1"
    "fig:6.2.2"
    "fig:6.2.1.a"
    "tab:7.2.1"
    "tab:8.2.1"
    "tab:8.2.2"
    "fig:9.1.1"
    "tab:10.1.1"))
 :latex)

