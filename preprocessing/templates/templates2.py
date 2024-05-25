basic = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

anttor = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math]{anttor}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

anttor_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[condensed, math]{anttor}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

anttor_light = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[light, math]{anttor}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

anttor_light_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[ligth, condensed, math]{anttor}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

arev = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{arev}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

gfsartemisia = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{gfsartemisia}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

gfsartemisia_euler = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{gfsartemisia-euler}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

baskervald_x = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[lf]{Baskervaldx}
\usepackage[bigdelims,vvarbb]{newtxmath}
\usepackage[cal=boondoxo]{mathalfa}
\renewcommand*\oldstylenums[1]{\textosf{#1}}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

baskervillef = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{baskervillef}
\usepackage[varqu,varl,var0]{inconsolata}
\usepackage[scale=.95,type1]{cabin}
\usepackage[baskerville,vvarbb]{newtxmath}
\usepackage[cal=boondoxo]{mathalfa}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

boisik = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{boisik}
\usepackage[OT1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

bitstream_charter = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[bitstream-charter]{mathdesign}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

concmath = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{concmath}
\usepackage[OT1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

european_concmath = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{concmath}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

concmath_euler = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{beton}
\usepackage{euler}
\usepackage[OT1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

computer_modern = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[OT1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

computer_modern_bright = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{cmbright}
\usepackage[OT1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

european_computer_modern_bright = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{cmbright}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

european_computer_modern = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

european_computer_modern_sans_serife = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{sansmathfonts}
\usepackage[T1]{fontenc}
\renewcommand*\familydefault{\sfdefault}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

domitian = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathpazo}
\usepackage{domitian}
\usepackage[T1]{fontenc}
\let\oldstylenums\oldstyle

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

drm = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{drm}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

erewhon = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[proportional,scaled=1.064]{erewhon}
\usepackage[erewhon,vvarbb,bigdelims]{newtxmath}
\usepackage[T1]{fontenc}
\renewcommand*\oldstylenums[1]{\textosf{#1}}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

erewhon_cc_euler_math = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{beton}
\usepackage{euler}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

eb_garamond = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[cmintegrals,cmbraces]{newtxmath}
\usepackage{ebgaramond-maths}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

heuristica = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{heuristica}
\usepackage[heuristica,vvarbb,bigdelims]{newtxmath}
\usepackage[T1]{fontenc}
\renewcommand*\oldstylenums[1]{\textosf{#1}}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

iwona = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math]{iwona}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

iwona_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[condensed,math]{iwona}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

iwona_light = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[light,math]{iwona}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

iwona_light_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[light,condensed,math]{iwona}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kerkis = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{kmath,kerkis}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kp_sans_serif = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[sfmath]{kpfonts}
\renewcommand*\familydefault{\sfdefault} 
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kp_serif = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{kpfonts}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kurier = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[math]{kurier}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kurier_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[condensed,math]{kurier}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kurier_light = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[light,math]{kurier}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

kurier_light_condensed = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[light,condensed,math]{kurier}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

latin_modern_roman = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{lmodern}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

libertinus_serif = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{libertinus}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

linux_libertine = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{libertine}
\usepackage{libertinust1math}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

lx_fonts = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{lxfonts}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

mlmodern = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mlmodern}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

gfs = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[default]{gfsneohellenic}
\usepackage[LGR,T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

new_px = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{newpxtext,newpxmath}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

new_px_euler = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{newpxtext,eulerpx}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

new_tx = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

urw_nimbus_roman = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathptmx}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

noto_serif = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{notomath}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

urw_palladio = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[sc]{mathpazo}
\linespread{1.05}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

px_fonts = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pxfonts}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

scholax = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[p,osf]{scholax}
\usepackage{amsmath,amsthm}
\usepackage[scaled=1.075,ncf,vvarbb]{newtxmath}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

urw_schoolbook_l = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fouriernc}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

step = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[notext]{stix}
\usepackage{step}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

sticks_too = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{stickstootext}
\usepackage[stickstoo,vvarbb]{newtxmath}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

stix = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{stix}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

stix2 = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\usepackage{stix2}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

tx_fonts = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{txfonts}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

utopia_fourier = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fourier}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""

utopia_math = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[adobe-utopia]{mathdesign}
\usepackage[T1]{fontenc}

\begin{document}

\begin{align*}
%s
\end{align*}

\end{document}
"""