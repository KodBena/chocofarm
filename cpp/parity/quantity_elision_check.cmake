# cpp/parity/quantity_elision_check.cmake
# Purpose: the executable half of the Quantity<Tag,Rep> ZERO-COST elision A/B gate (cpp/CMakeLists.txt
#   target chocofarm-quantity-elision-check). Given the two object files (the SAME hot kernel compiled over
#   the strong type vs the bare rep, -O3 -march=native), objdump-disassembles the `hot` symbol in each,
#   strips the address-column noise, and asserts the two instruction sequences are BYTE-IDENTICAL. A
#   mismatch is a FATAL_ERROR (fails the build) — the phantom is not zero-cost, which blocks everything
#   (ADR-0002 fail loud; ADR-0009 the perf claim carries its runnable substantiation).
#
# Public Domain (The Unlicense).
find_program(OBJDUMP objdump REQUIRED)

function(disasm_hot OBJ OUTVAR)
  execute_process(
    COMMAND ${OBJDUMP} -d --no-show-raw-insn ${OBJ}
    OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "objdump failed on ${OBJ}")
  endif()
  # keep only the <hot>: block; drop the leading address column ("  12:\t") so only the mnemonics remain
  string(REGEX MATCH "<hot>:\n([^\n]*\n)*" _block "${_raw}")
  string(REGEX REPLACE "[ \t]*[0-9a-f]+:\t" "" _block "${_block}")
  set(${OUTVAR} "${_block}" PARENT_SCOPE)
endfunction()

disasm_hot("${TYPED_OBJ}" TYPED_ASM)
disasm_hot("${RAW_OBJ}" RAW_ASM)

if(TYPED_ASM STREQUAL RAW_ASM)
  message(STATUS "Quantity elision gate: PASS — `hot` is BYTE-IDENTICAL over Quantity<Tag,uint32_t> and raw uint32_t (-O3 -march=native). Zero-cost.")
else()
  message("==== TYPED `hot` ====\n${TYPED_ASM}")
  message("==== RAW `hot` ====\n${RAW_ASM}")
  message(FATAL_ERROR "Quantity elision gate: FAIL — the strong-type and raw `hot` disassemblies DIFFER. The phantom is NOT zero-cost; this blocks everything (ADR-0009 / ADR-0000).")
endif()
