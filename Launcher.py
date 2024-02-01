
import gradio as Framework

import UI as UI

import os

Title = F"Line to Screentone"

Address_IP_Default = F"0.0.0.0"
Port_Default = 7860

class Application:

    def Launch(self, Address_In, Port_In, IsSharing_In = False):

        with Framework.Blocks() as Page:
            Page_main = self.Create_Page()
        return Page.queue().launch\
        (
            share = IsSharing_In,
            server_name = Address_In,
            server_port = Port_In,
        )

    def Create_Page(self):

        Framework.Markdown(F"<h1>{Title}<h1>")

        Tab_Screentone = UI.Tab()


def Execute_Application\
(
    Address_In = Address_IP_Default,
    Port_In = Port_Default,
    IsSharing_In = False
):
    Application().Launch\
    (
        Address_In = Address_In,
        Port_In = Port_In,
        IsSharing_In = IsSharing_In,
    )

if __name__ == F"__main__":
    Execute_Application\
    (
        Address_In = "localhost",
        Port_In = 2222,
        IsSharing_In = False,
    )
else:
    pass
