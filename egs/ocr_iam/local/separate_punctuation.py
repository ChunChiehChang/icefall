#!/usr/bin/env python3

import sys
import re

for line in sys.stdin:
    line = re.sub(r'(^|[\s])(?P<punctuation>[!(),.?;:\'"])(?P<letter>[\w])', r' \g<punctuation> \g<letter>', line)
    line = re.sub(r'(?P<letter>[\w])(?P<punctuation>[!(),.?;:\'"])[\s]', r'\g<letter> \g<punctuation> ', line)
    sys.stdout.write(line.strip() + '\n')
