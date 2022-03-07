kill -15 $(ps aux | grep lxx22 | grep ray | grep compose | grep -v grep | awk '{print $2}')
# kill -15 $(ps aux | grep lxx22 | grep nvprof | grep -v grep | awk '{print $2}')