file(GLOB panoc_SRC
    "*.c"
)
file(GLOB constraint_SRC
    "constraints/*.c"
)

message("hey ${constraint_SRC}")
add_library(PANOC_lib STATIC foo.c ${panoc_SRC} ${constraint_SRC})