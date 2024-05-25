basic = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}

\begin{displaymath}
%s
\end{displaymath}

\end{document}
"""

template_1 = r"""
\documentclass[11pt]{article}
\usepackage[]{acl}
\usepackage{times}
\usepackage{times}
\usepackage{latexsym}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{threeparttable}
\newtheorem{theorem}{Theorem}
\usepackage{enumitem}
\usepackage{float}
\usepackage{subcaption}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\begin{document}

$$
%s
$$

\end{document}
"""

template_2 = r"""
\documentclass[aps,prd,reprint, amsmath,amssymb]{revtex4-2}
\usepackage{acro} 
\usepackage{graphicx}
\usepackage{dcolumn}
\usepackage{bm}
\usepackage{hyperref}
\usepackage{float}
\usepackage{enumitem}
\pagenumbering{gobble}
\begin{document}
$$
%s
$$
\end{document}
"""

template_3 = r"""
\documentclass[aps,prl,reprint,longbibliography]{revtex4-1}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{algpseudocode}
\usepackage{algorithm}
\usepackage{enumitem}
\usepackage{float}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{thm-restate}
\pagenumbering{gobble}
\begin{document}
$$
%s
$$
\end{document}
"""

template_4 = r"""
\documentclass[preprint,12pt]{elsarticle}

\usepackage{amssymb}
\usepackage{times}
\usepackage{soul}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[utf8]{inputenc}
\usepackage[small]{caption}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{multirow}
\pagenumbering{gobble}
\journal{Neurocomputing}
\begin{document}
$$
%s
$$
\end{document}
"""

template_5 = r"""
\documentclass[a4paper,10pt]{amsart}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{relsize}
\usepackage{amssymb,amsmath}
\usepackage{dsfont}
\usepackage[all]{xy}
\pagenumbering{gobble}
\input{macros.tex}
\begin{document}
$$
%s
$$
\end{document}
"""

template_6 = r"""
\documentclass[twocolumn]{aastex631}
\pagenumbering{gobble}
\begin{document}
$$
%s
$$
\end{document}
"""