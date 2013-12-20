find . -name '*_test' | while read file; do
    if [[ -x "$file" ]]
    then
        echo "Running test:" $file;
        $file;
    fi
done
