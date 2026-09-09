: "${GCOV=gcov}"
make clean
make 'CFLAGS = -coverage -Og -g3' 'LDFLAGS = -coverage' &&
    sh run.sh &&
    gcovr --gcov-executable "$GCOV" --gcov-suspicious-hits-threshold 0 \
          --html-details cover.html
