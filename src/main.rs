use moebius::app::App;

fn main() {
    console_error_panic_hook::set_once();

    if !has_browser_body() {
        return;
    }

    leptos::mount::mount_to_body(App);
}

#[inline]
fn has_browser_body() -> bool {
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.body())
        .is_some()
}
