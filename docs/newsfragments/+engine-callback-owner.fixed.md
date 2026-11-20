The Rust engine session now keeps its callback state behind a raw pointer that
it frees on drop or replacement, so the context pointer that native code stores
stays valid when the session moves the state. Previously the pointer was
derived from a box that was then moved, which is undefined behavior under the
Rust aliasing rules.
