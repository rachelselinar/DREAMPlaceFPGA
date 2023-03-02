set outfile [open "io_placement.txt" w]

set ios [get_cells -hierarchical -filter { PRIMITIVE_TYPE =~ I/O.*.* || PRIMITIVE_TYPE == CLOCK.BUFFER.BUFGCE } ]

foreach io $ios {
    puts -nonewline $outfile $io
    puts -nonewline $outfile " "
    puts $outfile [get_property SITE $io]
}

close $outfile