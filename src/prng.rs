/// Tiny xoshiro256** -style PRNG used for weight init and shuffling.
pub struct Prng {
    s: [u64; 4],
}

impl Prng {
    pub fn new(seed0: u64, seed1: u64) -> Self {
        let mut rng = Prng {
            s: [seed0, seed1, seed0 ^ 0x9e3779b97f4a7c15, seed1 ^ 0xbf58476d1ce4e5b9],
        };
        // warm up a bit so short seeds still mix
        for _ in 0..16 {
            let _ = rng.next_u64();
        }
        rng
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Uniform in [0, 1).
    pub fn randf(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

thread_local! {
    static GLOBAL: std::cell::RefCell<Prng> =
        std::cell::RefCell::new(Prng::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb));
}

/// Global RNG used for training-order shuffles.
pub fn rand() -> u64 {
    GLOBAL.with(|g| g.borrow_mut().next_u64())
}
