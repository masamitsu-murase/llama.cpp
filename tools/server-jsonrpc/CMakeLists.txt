include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET llama-server-jsonrpc)

add_executable(${TARGET}
    main.cpp
    jsonrpc.cpp
    jsonrpc.h
)

target_include_directories(${TARGET} PRIVATE ../server ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
