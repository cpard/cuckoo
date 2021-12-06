# Cuckoo Hashing

This is an implementation in Rust of a d-ary [cuckoo hashing](https://en.wikipedia.org/wiki/Cuckoo_hashing) with stash. This implementation is based on the implementation done by [Utkan Güngördü](https://github.com/salviati) in Golang. You can find the implementation [here](https://github.com/salviati/cuckoo).

There are a couple of differences in this implementation, with the most notable one being the use of bitmaps instead of using zero as a placeholder for empty cells in the map. What Uktan did with zero is smart but I thought it would be a good exercise to  build it without the trick, there's probably a small performance penalty but that's fine.

I did this as an exercise in Rust, this structure is a good one to get deeper into Rust but without having to deal with the complexity of linked lists. It was a great opportunity to understand better things like types and struct and of course to get used to the borrow checker.

My experience so far is that if you want to learn the language, it's better to consider the compiler as a teacher. Start a project and listen to the compiler as you try to make it compile.

A hashmap based on Cuckoo Hashing is something covered a lot under database courses. You can find plenty of information about it on the web, below you can check some of the resources I used for this project.

[CMU Intro to Database Systems Fall 2021](https://15445.courses.cs.cmu.edu/fall2021/slides/06-hashtables.pdf)

[libcuckoo](https://github.com/efficient/libcuckoo) in case you are interested in a high performance implementation

[A guide to Cuckoo Hashing](https://programming.guide/cuckoo-hashing.html)

You can also take a look to the source code, if you want to implement Cuckoo Hashing as a way to learn using Rust, I'd suggest to try and follow the Golang implementation as I did and take a look at this implementation if you feel stuck on something.

I'm pretty sure most of the things I did, could have been implemented in a better way, especially a number of clone operations that can be found in the code. If you have any suggestions on how the code can be improved, please let me know!

TODO: Implement iterators to iterate over keys and (k,v). This will be a good exercise in implementing iterators in Rust.
