extern crate simple_dcgan;

use simple_dcgan::Trainer;

fn main() {
    let mut trainer = Trainer::new("config.json");
    trainer.train();
}