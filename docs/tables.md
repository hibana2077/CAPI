# Tables

`grep best ../../SW{001..036}.log | awk -F':' '{print $1, $NF}' | sort -k2 -n | tail -n 1`