fn="situation_current"
plantuml ./$fn.pu -tsvg -nometadata
/usr/bin/env python postprocess_svg.py $fn.svg --links_color '#0000f0' --ignore_links="3D augmentation"
#rsvg-convert -o $fn.png $fn.svg -w 800 --keep-aspect-ratio

