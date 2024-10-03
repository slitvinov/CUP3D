make 'CXXFLAGS = -coverage -Og -g3' 'LDFLAGS = -coverage' &&
    sh run.sh &&
    python -m gcovr --html-details cover.html
